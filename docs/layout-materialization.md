# Layout: TritonGPU에서 결정하고, 아래는 번역만 한다

상태: 구현 완료 (2026-10-06). 스위프 514/514. 옛 경로(perm 문자열 해석, t01~t04, t06, 첫 판
materialize-layouts)는 지웠다.

## 1. 원칙

- layout은 TritonGPU(TTGIR)에서 **결정**한다. anchor 한 패스 + `remove-layout-conversions`(RLC)
  한 번이다.
- 그 아래(triton_shared, psto stage 4)는 결정하지 않는다. 위의 layout을 기계 모양으로 옮겨 적기만
  한다.
- 데이터가 레인 사이를 옮겨 가는 곳은 TTGIR의 `ttg.convert_layout`뿐이다.
- 모양 변환(reshape, transpose, collapse, expand, split, cat)은 layout을 **합성**한다. 원소는 같은
  자리에 있고 좌표의 이름만 바뀐다(`L' = T ∘ L`).

## 2. 기계 대응

| Triton (GPU) | 이 기계 |
|---|---|
| 스레드 | 레인 (bank 하나) |
| 레지스터 | bank 안의 깊이 |
| warp, CTA | 1 |
| warp shuffle | XLU (레인 사이 이동) |
| 레인 기저 0 | 모든 레인이 사본을 든다 |

## 3. 패스별 역할

| 단계 | 패스 | 하는 일 |
|---|---|---|
| TTGIR | `convert-triton-to-tritongpu` | 기본 blocked layout (warp 1, 레인 256) |
| | `tritongpu-torchsim-anchor-layouts` | 기계가 강제하는 layout을 한 번에 적는다 (4절) |
| | `tritongpu-remove-layout-conversions` | upstream 전파. 한 번만 돈다 |
| | `tritongpu-torchsim-strip-pins` | anchor의 pin(항등 gather)을 원본으로 되돌린다 → `02h-layouts.mlir` |
| triton_shared | `TTSHARED_PIPELINE` | encoding을 실은 채 linalg/tensor로 내린다 |
| | `tritongpu-torchsim-regularize-layouts` | layout 정규화: 모든 encoding을 그 값 모양의 `#ttg.linear`로 적고(0인 레지스터 기저는 뺀다), upstream의 크기 1 축 접기를 돌린다. 접힌 값의 layout은 이름만 바꾼 합성이고, rank 0까지는 접지 않는다 → `03-ttshared.mlir` |
| stage 4 | p00 | 지원 안 하는 op 변환. 긴 scan 접기는 layout 합성으로 |
| | p01 | 접근을 transfer로. index와 mask는 원소와 같은 layout |
| | p02 materialize-layouts | layout을 물리 모양으로 번역하고 encoding을 지운다 (5절) |
| | p03 이하 | layout을 모른다 |

## 4. anchor

`TorchsimAnchorLayouts.cpp`. 결정은 모두 여기서 한다.

| op | 적는 layout | 이유 |
|---|---|---|
| `tt.dot` | A, B, C 각각 자기 모양의 linear layout. 레인은 마지막 축. 레인보다 긴 축의 윗비트는 가장 느린 레지스터 | systolic array가 세 피연산자 모두를 고정한다 |
| `tt.reduce` | 레인이 축을 피하면 그대로, 아니면 축이 아닌 넓은 축으로, 없으면 한 bank. 피연산자에 pin | 아래의 reduce는 레인을 건너지 않는다 |
| 긴 `tt.scan` | 원소 i가 레인 i/lanes, 레지스터 i%lanes | p00의 접기가 이름 바꾸기가 된다 |
| `tt.reshape allow_reorder` | 원소가 있는 자리 그대로 | 순서가 자유이므로 이동이 필요 없다 |
| `tt.cat` | lhs 다음 rhs (triton_shared가 이어 붙이는 순서). 결과에 pin | TTGIR 기본값(번갈아)과 triton_shared의 순서가 다르다 |

pin은 `efficient_layout`이 붙은 항등 `tt.gather`다. RLC가 그 layout을 바꾸지 못하게 한다.

## 5. 물리 모양 (p02 materialize-layouts)

`lib_linear_layout.physical_perm`이 정한다. 물리 모양은 `[레인 축, 깊이...]`다.

- **레인 축**: 레인 기저가 걸린 축. 레인보다 길면 윗비트가 가장 느린 레지스터에 있다(wrap).
- **깊이**: 레지스터 기저를 느린 것부터 늘어놓은 순서.
- **레인이 없는 값**(사본): 크기 1 축이 한 bank를 대표한다. 그런 축이 없으면 digit 단계가 lead
  축을 붙인다.
- **물리적으로 표현할 수 없는 배치**: 한 축이 여러 run으로 쪼개져 있으면 digit으로 나눈다.
  레인이 여러 축에 걸쳐 있으면 레인 비트가 가장 많은 축 하나로 옮긴다(`lanes_on_one_dim`). 그
  값을 convert만 읽으면 reshape이 convert의 layout을 바로 낸다. 이동은 한 번이다.

사본은 **넓히는 broadcast가 읽는 값에만** 물리 모양에 넣는다. 만드는 op마다 사본을 채우는 법이
정해져 있다.

| 만드는 op | 사본을 채우는 법 |
|---|---|
| empty, fill, 인덱스 산술 (generic) | 레인마다 다시 계산한다 |
| load | DMA stride 0. 한 행을 모든 레인에 쓴다 |
| reduce | 입력을 넓혀 bank 안에서 다시 줄인다 |
| transpose, collapse, expand, cast | 이름을 바꿔 원본을 다시 짓는다 |
| `ttg.convert_layout` | 원본도 사본이면 원본을 다시 짓는다. 아니면 broadcast 하나 → p07이 XLU all-gather로 내린다 |
| 그 밖 | 거부 (`DigitRefused`) |

이동은 convert 자리에만 `torchsim.copy`로 적는다. 어떤 XLU 연산을 쓸지는 p07이 고른다.

## 6. 측정 (2026-10-06)

옛 경로 산출물 `sweep-old3`과 같은 496개 커널 기준이다.

| | 옛 경로 | 새 경로 |
|---|---|---|
| 통과 | 513/513 | 514/514 (커버리지 커널 하나 추가) |
| XLU 이동 | 674 | 346 (permute 161, crossing 116, all-gather 70, replicate 0) |
| 옛 경로보다 이동이 많은 커널 | — | 0 |

expect_ir pin 중 옛 경로를 잰 것은 새 주장으로 바꿨다. 예를 들어 reshape 커널은 이제
"reshape은 아무것도 옮기지 않는다"를 pin한다.

## 7. 첫 판(TTGIR 끝 물리화)을 버린 이유

- encoding이 지워지면 TTGIR에서 구별되던 값이 하나가 된다. 두 `arange`가 CSE로 합쳐지고,
  하류는 그 인덱스를 데이터처럼 옮겼다.
- 인덱스 사슬을 물리 모양으로 다시 지어야 했다. 그 때문에 MaskAnalysis가 넓게 태어난 i1을 읽지
  못했고, 그 수정은 되돌렸다.

## 8. 남은 질문

- 레인보다 긴 축의 permute(`lib_cross_lane`의 접기 경로)는 이제 어느 커널에서도 나오지 않는다.
  긴 축은 digit으로 쪼개지거나 wrap이 되어 crossing으로 간다. 닿지 않는 경로인지 확인한 뒤
  지울지 정해야 한다.

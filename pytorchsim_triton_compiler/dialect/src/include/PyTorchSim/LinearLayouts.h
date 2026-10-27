// TritonGPU's LinearLayout as the passes in Python ask it, through _pytorchsim_ext: bases in, bases out.
#ifndef PYTORCHSIM_LINEARLAYOUTS_H
#define PYTORCHSIM_LINEARLAYOUTS_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pytorchsim {

/// Each in-dim's name and bases, in order: a vector per basis, one int per out-dim.
using NamedBases = std::vector<std::pair<std::string, std::vector<std::vector<int32_t>>>>;

/// `bases` over a tile of `src` after a row-major reshape to `dst`, as Triton's reshapeLayout gives
/// them; empty `why` on success, else the reason it refuses (a shape not of powers of two, among them).
NamedBases reshapeBases(const NamedBases &bases, const std::vector<int64_t> &src,
                        const std::vector<int64_t> &dst, std::string &why);

} // namespace pytorchsim

#endif // PYTORCHSIM_LINEARLAYOUTS_H

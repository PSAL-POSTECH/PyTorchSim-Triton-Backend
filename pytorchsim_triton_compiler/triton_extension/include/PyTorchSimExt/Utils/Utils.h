//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef PYTORCHSIM_EXT_UTILS_UTILS_H
#define PYTORCHSIM_EXT_UTILS_UTILS_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);

// The number of bytes one element of `t` occupies. A sub-byte type still owns a
// byte: a torch.bool buffer arrives as !tt.ptr<i1> and is stepped one byte at a
// time, which is the whole reason triton puts a bitcast in front of it.
unsigned storageBytes(Type t);

// How many DST elements one SRC element spans across this pointer bitcast, or 0
// when no whole number does. Offsets are counted in ELEMENTS, so crossing the
// cast means multiplying every one of them by this; 1 is the address-preserving
// case. Both walks that cross a pointer chain ask this same question.
unsigned ptrCastElementRatio(BitcastOp op);

// Does `(offset + p*stride) mod wrap` never fire, for p in [0, size)? A wrap
// that cannot fire is not a fact about the address, and both the analysis that
// would record one and the verifier that forbids one ask this one question.
// False whenever any of the four is not a compile-time constant -- which is
// also what keeps a loop-carried offset out, since that one is an iter-arg.
bool wrapNeverFires(OpFoldResult wrap, OpFoldResult offset, OpFoldResult stride,
                    int64_t size);

// One number out of a constant: a scalar, or a tensor whose elements are all
// the same. `getConstantIntValue` answers only the first, and an address term
// is a tensor as often as it is a scalar.
std::optional<int64_t> constantIntOrSplat(Value v);
} // namespace triton

} // namespace mlir

#endif // PYTORCHSIM_EXT_UTILS_UTILS_H

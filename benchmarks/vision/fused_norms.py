"""Inference-only Triton normalization candidates, preserving Qwen's BF16 cast order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, G, Y, N: tl.constexpr, EPS: tl.constexpr,
          GATED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets, offsets < N, 0).to(tl.float32)
    weight = tl.load(W + offsets, offsets < N, 0).to(tl.float32)
    normalized = values * tl.rsqrt(tl.sum(values * values, 0) / N + EPS)
    if GATED:
        # Qwen casts the norm to input dtype, multiplies by a same-dtype
        # weight, then applies an fp32 SiLU gate and casts back.
        normalized = normalized.to(X.dtype.element_ty).to(tl.float32)
        normalized = (normalized * weight).to(X.dtype.element_ty).to(tl.float32)
        gate = tl.load(G + row * N + offsets, offsets < N, 0).to(tl.float32)
        normalized = normalized * (gate / (1.0 + tl.exp(-gate)))
    else:
        normalized = normalized * (1.0 + weight)
    tl.store(Y + row * N + offsets, normalized, offsets < N)


def install_norm_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    def plain(self, x):
        x = x.contiguous()
        output = torch.empty_like(x)
        n = x.shape[-1]
        _norm[(x.numel() // n,)](x, self.weight, x, output, n, self.eps,
                               False, triton.next_power_of_2(n))
        return output

    def gated(self, x, gate):
        x, gate = x.contiguous(), gate.contiguous()
        output = torch.empty_like(x)
        n = x.shape[-1]
        _norm[(x.numel() // n,)](x, self.weight, gate, output, n, self.variance_epsilon,
                               True, triton.next_power_of_2(n))
        return output

    qwen.Qwen3_5RMSNorm.forward = plain
    qwen.Qwen3_5RMSNormGated.forward = gated


def verify_norms():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    reference, reference_gated = qwen.Qwen3_5RMSNorm.forward, qwen.Qwen3_5RMSNormGated.forward
    install_norm_kernels()
    results = []
    with torch.inference_mode():
        for size in (128, 5120):
            model = qwen.Qwen3_5RMSNorm(size).cuda().bfloat16()
            model.weight.uniform_(-0.1, 0.1)
            x = torch.randn((32, size), device="cuda", dtype=torch.bfloat16)
            expected, actual = reference(model, x), model(x)
            torch.testing.assert_close(expected, actual, atol=0.005, rtol=0.01)
            results.append({"kernel": "rmsnorm", "size": size,
                            "maxError": float((expected.float() - actual.float()).abs().max())})
        model = qwen.Qwen3_5RMSNormGated(128).cuda().bfloat16()
        x, g = [torch.randn((32, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        expected, actual = reference_gated(model, x, g), model(x, g)
        torch.testing.assert_close(expected, actual, atol=0.005, rtol=0.01)
        results.append({"kernel": "rmsnorm_gated", "size": 128,
                        "maxError": float((expected.float() - actual.float()).abs().max())})
    return results

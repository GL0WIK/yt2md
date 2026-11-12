"""Candidate adapter, used only by the separate benchmark process."""


def install_convolution_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from fla.modules.conv.triton import causal_conv1d_fwd, causal_conv1d_update

    def prefill(hidden_states, weight, bias=None, activation=None, **kwargs):
        output, _ = causal_conv1d_fwd(hidden_states.transpose(1, 2), weight, bias, None,
                                     output_final_state=False, activation=activation)
        return output.transpose(1, 2)

    def decode(hidden_states, conv_state, weight, bias=None, activation=None):
        output, _ = causal_conv1d_update(hidden_states.transpose(1, 2), conv_state,
                                        weight=weight, bias=bias, activation=activation)
        return output.transpose(1, 2)

    qwen.causal_conv1d_fn = prefill
    qwen.causal_conv1d_update = decode


def verify_numerical_equivalence():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    torch.manual_seed(123)
    results = []
    with torch.inference_mode():
        for length, reference, accelerated in [
            (1, qwen.torch_recurrent_gated_delta_rule.__wrapped__, fused_recurrent_gated_delta_rule),
            (128, qwen.torch_chunk_gated_delta_rule.__wrapped__, chunk_gated_delta_rule),
        ]:
            shape = (1, length, 4, 128)
            query, key, value = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(3)]
            gate = -torch.rand((1, length, 4), device="cuda") * 0.1
            beta = torch.rand((1, length, 4), device="cuda", dtype=torch.bfloat16)
            state = torch.randn((1, 4, 128, 128), device="cuda") * 0.01
            expected = reference(query, key, value, g=gate, beta=beta, initial_state=state.clone(),
                                 output_final_state=True, use_qk_l2norm_in_kernel=True)
            actual = accelerated(query, key, value, g=gate, beta=beta, initial_state=state.clone(),
                                 output_final_state=True, use_qk_l2norm_in_kernel=True)
            for label, ref, out in zip(("output", "state"), expected, actual):
                torch.testing.assert_close(ref.float(), out.float(), atol=2e-3, rtol=2e-2)
                results.append({"kernel": "delta", "length": length, "tensor": label,
                                "maxError": float((ref.float() - out.float()).abs().max())})
        reference_prefill, reference_decode = qwen.causal_conv1d_fn.__wrapped__, qwen.causal_conv1d_update.__wrapped__
        install_convolution_kernels()
        weight = torch.randn((256, 4), device="cuda", dtype=torch.bfloat16) * 0.1
        for length in (1, 128):
            hidden = torch.randn((1, 256, length), device="cuda", dtype=torch.bfloat16)
            expected = reference_prefill(hidden, weight, activation="silu")
            actual = qwen.causal_conv1d_fn(hidden, weight, activation="silu")
            torch.testing.assert_close(expected, actual, atol=2e-3, rtol=2e-2)
            results.append({"kernel": "conv_prefill", "length": length,
                            "maxError": float((expected.float() - actual.float()).abs().max())})
        state = torch.randn((1, 256, 4), device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn((1, 256, 1), device="cuda", dtype=torch.bfloat16)
        expected_state, actual_state = state.clone(), state.clone()
        expected = reference_decode(hidden, expected_state, weight, activation="silu")
        actual = qwen.causal_conv1d_update(hidden, actual_state, weight, activation="silu")
        torch.testing.assert_close(expected, actual, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(expected_state, actual_state, atol=0, rtol=0)
        results.append({"kernel": "conv_decode", "length": 1,
                        "maxError": float((expected.float() - actual.float()).abs().max())})
    return results

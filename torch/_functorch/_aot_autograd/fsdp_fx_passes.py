import torch
import operator

"""
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]  ===== Forward graph 0 =====
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]  /data/users/willfeng/pytorch_yf225/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]     def forward(self, primals_1: "f32[152275600]", primals_2: "f32[12340]", primals_3, primals_4: "f32[2, 12340]", primals_5: "f32[152275600]", primals_6: "f32[12340]"):
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         # No stacktrace found for following nodes
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         clone: "f32[152275600]" = torch.ops.aten.clone.default(primals_1)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         clone_1: "f32[12340]" = torch.ops.aten.clone.default(primals_2)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         as_strided: "f32[152275600]" = torch.ops.aten.as_strided.default(clone, [152275600], [1], 0)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         as_strided_1: "f32[12340]" = torch.ops.aten.as_strided.default(clone_1, [12340], [1], 0)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_param.py:426 in unsafe_alloc_storage, code: tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(as_strided, 609102400)
[rank0]:I2024-03-02 17:49:03,982.982000 140133037143872 torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py:315] [0/0] [__aot_graphs]         resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(as_strided_1, 49360)
"""


def insert_primal_resize_to_full_at_start_and_resize_to_0_at_end_of_graph(mod):
    # NOTE: this is hacky, and only safe if the primal stays size-0 before and after the graph (i.e. FSDP params)
    # Proactively resize some primal tensors to their full size before clone at beginning of graph, and then resize them back to 0 at end of graph.
    # Primal tensors being resized are only the ones that will be resized anyway after the clone->as_strided op chain added by AOTAutograd functionalization.
    # (See "Forward graph 0" above)
    # Doing resize to full proactively before clone, so that the clone won't fail.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    primal_inputs_resized = set()
    for n in mod.graph.nodes:
        # Super complicated way to know that this is a size-0 primal input being cloned at beginning of graph.
        if (
            n.target is torch.ops.inductor.resize_storage_bytes_.default \
            and n.args[0].target is torch.ops.aten.as_strided.default \
            and n.args[0].args[0].target is torch.ops.aten.clone.default \
            and n.args[0].args[0].args[0] in primal_inputs_tensor_only
        ):
            primal_inputs_resized.add(n.args[0].args[0].args[0])
    for primal_input in list(primal_inputs_resized):
        first_non_placeholder_op = None
        return_op = None
        for n in mod.graph.nodes:
            if n.op != "placeholder":
                first_non_placeholder_op = n
                break
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        assert first_non_placeholder_op is not None
        with mod.graph.inserting_before(first_non_placeholder_op):
            full_size = primal_input.meta['val'].untyped_storage().size()  # 'val' is fake tensor which always has full size
            assert full_size > 0
            primal_input_resized_to_full = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (primal_input, full_size), {})
            # primal_input.replace_all_uses_with(primal_input_resized_to_full, propagate_meta=True)
            # mod.graph.erase_node(primal_input)  # probably should not do this? not sure.
        with mod.graph.inserting_before(return_op):
            primal_input_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (primal_input, 0), {})
        mod.graph.lint()
        mod.recompile()


def if_tensor_is_resized_to_full_then_resize_it_to_0_at_end_of_graph(mod):
    # FSDP graph has this invariant that if a tensor needs to be resized to full during execution of the graph, it *will* be resized to 0 again before exit of graph.
    tensors_resized = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] > 0:
            tensors_resized.add(n.args[0])
    for tensor in list(tensors_resized):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (tensor, 0), {})
        mod.graph.lint()
        mod.recompile()


def move_resize_to_0_to_end_of_graph(mod):
    # This pass is always a good idea to do so to avoid any use-after-free issues.
    resize_to_0_nodes = set()
    for n in mod.graph.nodes:
        if n.target is torch.ops.inductor.resize_storage_bytes_.default and n.args[1] == 0:
            resize_to_0_nodes.add(n)
    for resize_to_0_node in list(resize_to_0_nodes):
        return_op = None
        for n in mod.graph.nodes:
            if n.op == "output":
                return_op = n
                break
        with mod.graph.inserting_before(return_op):
            tensor_resized_to_0 = mod.graph.call_function(torch.ops.inductor.resize_storage_bytes_.default, (resize_to_0_node.args[0], 0), {})
        mod.graph.erase_node(resize_to_0_node)
        mod.graph.lint()
        mod.recompile()


def replace_primal_clone_at_beginning_of_graph_with_primal(mod):
    # Replace `clone(primal)` at beginning of graph with `primal`.
    # This is only safe if the graph does not have any autograd-affecting mutations and not explicitly cloning the primal through user code.
    # (i.e. only `with no_grad(): foreach_copy_` and `resize_storage_bytes_` is supported now).
    # TODO add checks to make sure the above invariant is maintained.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if n.op != "placeholder" and n.target is not torch.ops.inductor.resize_storage_bytes_.default:
            if n.target is torch.ops.aten.clone.default:
                if n.args[0] in primal_inputs_tensor_only:
                    n.replace_all_uses_with(n.args[0])
                    mod.graph.erase_node(n)
                    mod.graph.lint()
                    mod.recompile()
            else:
                break


def replace_primal_noop_as_strided_with_primal(mod):
    # Replace `as_strided(primal, ...)` with `primal`, if the as_strided is a no-op based on size and stride info. Should be always safe to do.
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for n in list(mod.graph.nodes):
        if (
            n.target is torch.ops.aten.as_strided.default \
            and n.args[0] in primal_inputs_tensor_only \
            and n.meta['val'].shape == n.args[0].meta['val'].shape \
            and n.meta['val'].stride() == n.args[0].meta['val'].stride()
        ):
            n.replace_all_uses_with(n.args[0])
            mod.graph.erase_node(n)
            mod.graph.lint()
            mod.recompile()


def arg_equals_or_contains_input_node(arg, inp_n):
    if isinstance(arg, (list, tuple)):
        return any(arg_equals_or_contains_input_node(a, inp_n) for a in arg)
    else:
        return arg == inp_n


def input_is_used_in_other_ops(ops, inp_n, except_callback):
    for n in ops:
        if (not except_callback(n)) and any(arg_equals_or_contains_input_node(arg, inp_n) for arg in n.args):
            return True
    return False


def reinplace_foreach_copy_if_input_has_no_other_use_in_graph(mod):
    """
    _foreach_copy_1 = torch.ops.aten._foreach_copy.default([view_1, view_2, view_3, view_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    getitem_44: "f32[2, 76137800]" = _foreach_copy_1[0]
    getitem_45: "f32[2, 6170]" = _foreach_copy_1[1]
    getitem_46: "f32[2, 76137800]" = _foreach_copy_1[2]
    getitem_47: "f32[2, 6170]" = _foreach_copy_1[3];  _foreach_copy_1 = None

    ->

    _foreach_copy__1 = torch.ops.aten._foreach_copy_.default([view_1, view_2, view_3, view_4], [getitem_28, getitem_33, getitem_38, getitem_43]);  view_1 = view_2 = view_3 = view_4 = getitem_28 = getitem_33 = getitem_38 = getitem_43 = None
    """
    # TODO: maybe super slow, need optimization
    for n in list(mod.graph.nodes):
        if n.target is torch.ops.aten._foreach_copy.default:
            _foreach_copy_outplace_node = n
            if all(not input_is_used_in_other_ops(list(mod.graph.nodes), inp_n, except_callback=lambda n: n == _foreach_copy_outplace_node) for inp_n in _foreach_copy_outplace_node.args[0]):
                with mod.graph.inserting_before(_foreach_copy_outplace_node):
                    for i, arg in enumerate(_foreach_copy_outplace_node.args[0]):
                        copy_to = arg
                        copy_from = _foreach_copy_outplace_node.args[1][i]
                        # _foreach_copy_inplace_node = mod.graph.call_function(torch.ops.aten._foreach_copy_.default, _foreach_copy_outplace_node.args, {})
                        # NOTE: Inductor seems to fail when encountering `_foreach_copy_` op. Need more investigation.
                        mod.graph.call_function(torch.ops.aten.copy_.default, (copy_to, copy_from), {})
                for node in list(mod.graph.nodes):
                    if node.target is operator.getitem and node.args[0] == _foreach_copy_outplace_node:
                        node.replace_all_uses_with(_foreach_copy_outplace_node.args[0][node.args[1]])
                        mod.graph.erase_node(node)
                mod.graph.erase_node(_foreach_copy_outplace_node)
                mod.graph.lint()
                mod.recompile()


def replace_as_strided_scatter_with_primal_if_primal_has_no_other_use_after_this_op(mod):
    """
    as_strided_scatter_3: "f32[12340]" = torch.ops.aten.as_strided_scatter.default(primals_4, view_15, [12340], [1], 0);

    ->

    primals_4
    """
    primal_inputs_tensor_only = [x for x in list(filter(torch._functorch.partitioners._is_primal, mod.graph.nodes)) if isinstance(x.meta.get('val', None), torch.Tensor)]
    for i, n in enumerate(list(mod.graph.nodes)):
        if n.target is torch.ops.aten.as_strided_scatter.default:
            as_strided_scatter_node = n
            if as_strided_scatter_node.args[0] in primal_inputs_tensor_only:
                primal = as_strided_scatter_node.args[0]
                if (
                    primal.meta['val'].shape == as_strided_scatter_node.meta['val'].shape \
                    and primal.meta['val'].stride() == as_strided_scatter_node.meta['val'].stride() \
                    and not input_is_used_in_other_ops(list(mod.graph.nodes)[i+1:], primal, except_callback=lambda n: n.target is torch.ops.inductor.resize_storage_bytes_ and n.args[1] == 0)
                ):
                    as_strided_scatter_node.replace_all_uses_with(primal)
                    mod.graph.erase_node(as_strided_scatter_node)
                    mod.graph.lint()
                    mod.recompile()

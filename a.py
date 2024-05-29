import torch
from typing import List
import tinygrad


torch2tinygrad = {
    "relu": "relu",
    "add": "add",
}


def to_tinygrad(tensor: torch.Tensor):
    return tinygrad.Tensor(tensor.numpy())


totiystr = """
def to_tinygrad(tensor: torch.Tensor):
    return tinygrad.Tensor(tensor.numpy())
"""
currinp = 0
currvar = 0


def get_input_name():
    global currinp
    currinp += 1
    return f"inp_{currinp}"


def get_var_name():
    global currvar
    return f"tmp_{currinp}"


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    inputs = []
    vars = dict()
    methodstr = totiystr
    methodstr += "def compiled():\n    "

    currident = 0

    def indent():
        nonlocal currident
        currident += 1

    def dedent():
        nonlocal currident
        currident -= 1

    def newline():
        return "\n" + "    " * currident

    indent()
    out = None
    for n in gm.graph.nodes:
        print(vars)
        op, target, args, kwargs, name = n.op, n.target, n.args, n.kwargs, n.name
        print(op, target, args, kwargs, name)
        if op == "placeholder":
            # lowercase
            target = target.lower()
            tname = get_input_name()
            inputs.append(target)
            vars[name] = tname
            methodstr += "global " + tname + newline()
            methodstr += f"{tname} = to_tinygrad({tname}){newline()}"
        elif op == "call_method":
            tinyop = torch2tinygrad.get(target)
            assert tinyop is not None, f"Unsupported op {target}"

            tname = get_var_name()
            args = [str(arg) for arg in args]
            targs = [vars[arg] for arg in args]
            methodstr += (
                f'{tname} = {targs[0]}.{tinyop}({", ".join(targs[1:])}){newline()}'
            )

            vars[name] = tname
        elif op == "output":
            out = vars[str(args[0][0])]

            vars[out] = "out"

            methodstr += "out = " + out + newline()
        else:
            raise NotImplementedError(
                f"Unsupported: {op}, {target}, {args}, {kwargs}, {name}"
            )

    assert out is not None, "No output found"
    methodstr += "return out,"
    print(methodstr)
    glo = {}
    glo["to_tinygrad"] = to_tinygrad
    glo["tinygrad"] = tinygrad
    glo["torch"] = torch

    def exe(torch_inps):
        torch_inps = (
            torch_inps if isinstance(torch_inps, (list, tuple)) else [torch_inps]
        )
        for inp, inpt in zip(inputs, torch_inps):
            print(inp, inpt)
            glo[vars[inp]] = inpt

        glo.update({k: v for k, v in zip(inputs, torch_inps)})
        exec(methodstr, glo)
        return glo["compiled"]()

    return exe


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

    def forward(self, x):
        return x.relu().add(x)


model = SimpleNN()

compiled = torch.compile(model, backend=custom_backend)


t = torch.randn(10)

x = compiled(t)

x2 = model(t)

print(x.numpy(), x2.detach().numpy())

"""Sandboxed code cells: the agent's programs run here, and reach the robot only through `robo`.

What a cell is
--------------
The agent sends ``python(checklist, code)``. The code runs in a per-episode namespace that persists across cells (variables and functions
are reused) and is dropped at episode end. The namespace holds a ``robo`` object, sandbox facades of the whitelisted modules and
plain data; it never holds the policy, the observation or any simulator object.

How a cell runs
---------------
Each cell runs in its own worker thread, used strictly as a coroutine: at any moment either the agent thread or the cell thread
runs, never both. A ``robo`` call converts its arguments to plain data inside the cell, posts one request to the agent thread and
blocks. The agent thread carries it out, including any native motion, and resumes the cell with the result. A thread is used
instead of an AST rewrite into generators because robo calls may sit anywhere in the agent's call stack (helper functions,
comprehensions, key functions); a suspended thread keeps that whole stack intact without changing Python semantics.

Sandbox rules (checked before any code runs, then enforced at run time)
-----------------------------------------------------------------------
* Imports only through a guarded ``__import__`` returning facades of math, numpy (with numpy.linalg), json, itertools, functools,
  statistics and collections. A facade holds only reviewed functions and constants, never a real module, so no attribute path
  leads to ``sys``, ``os``, files, sockets or numpy's file and ctypes interfaces.
* No ``open``, ``eval``/``exec``/``compile``, ``globals``/``locals``/``vars``, ``getattr``-family, ``input`` or ``breakpoint``; no names
  starting with ``__``; no attribute names starting with ``_`` (all dunders), no frame, code, generator or traceback introspection
  (``gi_*``, ``f_*``, ``tb_*``, ...), no ``mro``, ``str.format`` (whose field syntax walks attributes) and no file-writing array methods;
  no attribute assignment.
* No ``class``, ``with``, ``async``, ``match``, ``finally`` or bare/BaseException ``except``. Together these guarantee that no agent code
  can run while a cell is being stopped, and that dropping the namespace runs no agent finalizer on the agent thread.
* Limits per cell: Python compute time (a thread-local trace function raises an exception the cell cannot catch; time spent waiting
  for robo calls or native execution does not count). Bulk constructors are bounded: ``range``, sequence repetition, integer powers
  and shifts, numpy array constructors and products, broadcasting arithmetic and itertools combinatorics; ``itertools.count`` and
  ``cycle`` and unbounded ``repeat`` are absent.

Known limitation: the time limit acts between Python bytecodes. A single long C-level operation inside the size bounds cannot be
pre-empted in process; the agent thread stops waiting after the limit plus a grace period, abandons that cell thread and starts the
next cell with a fresh namespace. Memory is bounded only through the size limits above.
"""
from __future__ import annotations

import ast
import builtins
import dis
import collections as _collections
import functools as _functools
import itertools as _itertools
import json as _json
import math as _math
import operator as _operator
import queue
import statistics as _statistics
import sys
import threading
import time
import traceback
import types

import numpy as _np

CODE_LIMIT = 20_000            # characters per cell
ITEM_LIMIT = 10_000_000        # items one range, repetition or combinatoric iterator may produce
ARRAY_LIMIT = 10_000_000       # elements of one numpy result
INT_BITS_LIMIT = 1 << 16       # bits of an integer built by **, *, << or pow
FLOP_LIMIT = 2_000_000_000     # multiply-adds of one numpy product
CUBIC_LIMIT = 1_000_000        # elements of one matrix given to an O(n^3) linalg routine
PLAIN_LIMIT = 200_000          # elements of one robo argument
STDOUT_LIMIT = 8_000           # printed characters kept per cell
VALUE_LIMIT = 4_000            # characters of the final-expression summary
STUCK_GRACE_S = 2.0            # extra wait before a cell that holds the interpreter is abandoned


class SandboxViolation(ValueError):
    """The cell uses a construct outside the sandbox; nothing was executed."""


class SandboxLimitError(Exception):
    """A cell exceeded a size bound of the sandbox; the cell may catch it."""


class RoboError(Exception):
    """Misuse of a robo method: invalid arguments or an unavailable Tool."""


class RoboFailure(Exception):
    """A robo goal failed. The exception's result holds what happened; check=False returns the result instead of raising."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class _CellStop(BaseException):
    """Ends a cell. Agent code can neither name nor catch it."""
    status = "stopped"


class CellTimeout(_CellStop):
    status = "timeout"


class CellCancelled(_CellStop):
    status = "cancelled"


# ----------------------------------------------------------------------------------------
# Static validation and rewriting
# ----------------------------------------------------------------------------------------

_ALLOWED_NODES = tuple(getattr(ast, name) for name in (
    "Module", "Expr", "Assign", "AugAssign", "AnnAssign", "For", "While", "If", "Raise", "Try",
    "Assert", "Import", "ImportFrom", "Global", "Nonlocal", "Pass", "Break", "Continue",
    "Return", "Delete", "FunctionDef", "Lambda", "arguments", "arg", "keyword", "alias",
    "ExceptHandler", "comprehension", "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "IfExp",
    "Dict", "Set", "List", "Tuple", "ListComp", "SetComp", "DictComp", "GeneratorExp", "Yield",
    "YieldFrom", "Compare", "Call", "FormattedValue", "JoinedStr", "Constant", "Attribute",
    "Subscript", "Starred", "Name", "Slice", "Load", "Store", "Del", "And", "Or", "Add", "Sub",
    "Mult", "MatMult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    "FloorDiv", "Invert", "Not", "UAdd", "USub", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is",
    "IsNot", "In", "NotIn"))
_NODE_HINTS = {"ClassDef": "class definitions", "With": "with statements",
               "AsyncWith": "async code", "AsyncFor": "async code", "AsyncFunctionDef": "async code",
               "Await": "async code", "Match": "match statements", "TryStar": "except*"}
_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "memoryview", "BaseException", "SystemExit",
    "KeyboardInterrupt", "GeneratorExit"})
_BLOCKED_ATTRIBUTES = frozenset({
    "mro", "format", "format_map", "ctypes", "tofile", "dump", "resize", "setflags", "elements"})
_BLOCKED_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_", "co_")
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.MatMult: "@", ast.Div: "/",
           ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
           ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&"}
_COMPARE = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
            ast.GtE: ">="}


def blocked_attribute(name: str) -> bool:
    return name.startswith(_BLOCKED_PREFIXES) or name in _BLOCKED_ATTRIBUTES


def _members(module) -> set:
    return {name for name in vars(module) if not name.startswith("_")}


class _Checker(ast.NodeVisitor):
    def __init__(self, facades):
        self.facades = facades

    def fail(self, node, message):
        line = getattr(node, "lineno", None)
        raise SandboxViolation(f"line {line}: {message}" if line else message)

    def name(self, node, value, kind="name"):
        if value is not None and value.startswith("__"):
            self.fail(node, f"{kind} {value!r} is not available in code cells")

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            kind = type(node).__name__
            self.fail(node, _NODE_HINTS.get(kind, kind) + " are not available in code cells")
        super().generic_visit(node)

    def visit_Name(self, node):
        self.name(node, node.id)
        if node.id in _FORBIDDEN_NAMES:
            self.fail(node, f"{node.id!r} is not available in code cells")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if blocked_attribute(node.attr):
            hint = "; use an f-string" if node.attr in ("format", "format_map") else ""
            self.fail(node, f"attribute {node.attr!r} is not available in code cells{hint}")
        if not isinstance(node.ctx, ast.Load):
            self.fail(node, "attribute assignment and deletion are not available in code cells")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.name(node, node.name, "function name")
        self.generic_visit(node)

    def visit_arg(self, node):
        self.name(node, node.arg, "argument name")
        self.generic_visit(node)

    def visit_keyword(self, node):
        self.name(node, node.arg, "keyword")
        self.generic_visit(node)

    def visit_Global(self, node):
        for value in node.names:
            self.name(node, value)
        self.generic_visit(node)

    visit_Nonlocal = visit_Global

    def visit_Try(self, node):
        if node.finalbody:
            self.fail(node, "finally is not available in code cells; handle errors with except")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.fail(node, "a bare except is not available; name the exception, e.g. Exception")
        self.name(node, node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in self.facades:
                self.fail(node, f"import of {alias.name!r} is not available; allowed: "
                          + ", ".join(sorted(self.facades)))
            self.name(node, alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level or node.module not in self.facades:
            self.fail(node, f"import from {node.module!r} is not available; allowed: "
                      + ", ".join(sorted(self.facades)))
        members = _members(self.facades[node.module])
        for alias in node.names:
            if alias.name == "*" or (alias.name not in members
                                     and node.module + "." + alias.name not in self.facades):
                self.fail(node, f"{node.module}.{alias.name} is not available in code cells")
            self.name(node, alias.asname)
        self.generic_visit(node)


class _Rewriter(ast.NodeTransformer):
    """Route arithmetic and single comparisons through the size guards."""

    @staticmethod
    def call(name, *args):
        return ast.Call(ast.Name(name, ast.Load()), list(args), [])

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return ast.copy_location(self.call("__sbx_binop__", ast.Constant(_BINARY[type(node.op)]),
                                           node.left, node.right), node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        op = ast.Constant(_BINARY[type(node.op)])
        if isinstance(node.target, ast.Name):
            current = ast.Name(node.target.id, ast.Load())
            node.value = self.call("__sbx_checked__", op, node.value, current)
        else:
            node.value = self.call("__sbx_checked__", op, node.value)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
            return node
        return ast.copy_location(self.call("__sbx_compare__", ast.Constant(_COMPARE[type(node.ops[0])]),
                                           node.left, node.comparators[0]), node)


def validate(code: str, facades: dict) -> types.CodeType:
    """Check a cell against the sandbox rules and compile it; raise SandboxViolation."""
    if not isinstance(code, str):
        raise SandboxViolation("code must be a string")
    if len(code) > CODE_LIMIT:
        raise SandboxViolation(f"a code cell is limited to {CODE_LIMIT} characters")
    try:
        tree = ast.parse(code, filename="<cell>", mode="exec")
    except SyntaxError as error:
        raise SandboxViolation(f"line {error.lineno}: syntax error: {error.msg}") from None
    _Checker(facades).visit(tree)
    tree = _Rewriter().visit(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(ast.Expr(_Rewriter.call("__sbx_value__", last.value)), last)
    ast.fix_missing_locations(tree)
    try:
        return compile(tree, "<cell>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise SandboxViolation(f"syntax error: {error}") from None


# ----------------------------------------------------------------------------------------
# Run-time size guards
# ----------------------------------------------------------------------------------------

_OPERATORS = {"+": _operator.add, "-": _operator.sub, "*": _operator.mul, "@": _operator.matmul,
              "/": _operator.truediv, "//": _operator.floordiv, "%": _operator.mod,
              "**": _operator.pow, "<<": _operator.lshift, ">>": _operator.rshift,
              "|": _operator.or_, "^": _operator.xor, "&": _operator.and_}
_IN_PLACE = {"+": _operator.iadd, "-": _operator.isub, "*": _operator.imul, "@": _operator.imatmul,
             "/": _operator.itruediv, "//": _operator.ifloordiv, "%": _operator.imod,
             "**": _operator.ipow, "<<": _operator.ilshift, ">>": _operator.irshift,
             "|": _operator.ior, "^": _operator.ixor, "&": _operator.iand}
_COMPARATORS = {"==": _operator.eq, "!=": _operator.ne, "<": _operator.lt, "<=": _operator.le,
                ">": _operator.gt, ">=": _operator.ge}
_UNKNOWN = object()


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _product(values) -> int:
    total = 1
    for value in values:
        total *= int(value)
        if total > ITEM_LIMIT * 1000:
            break
    return total


def _limit(message):
    raise SandboxLimitError(message)


def _check_size(count, what="array"):
    if count > ARRAY_LIMIT:
        _limit(f"{what} would have more than {ARRAY_LIMIT} elements")


def _matmul_cost(sa, sb):
    sa = tuple(sa) or (1,)
    sb = tuple(sb) or (1,)
    if len(sa) == 1:
        sa = (1,) + sa
    if len(sb) == 1:
        sb = sb + (1,)
    batch = _product(_np.broadcast_shapes(sa[:-2], sb[:-2]))
    return batch * sa[-2] * sa[-1] * sb[-1], batch * sa[-2] * sb[-1]


def _check_arrays(op, a, b):
    if not (isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray)):
        return
    try:
        if op == "@":
            flops, size = _matmul_cost(_np.shape(a), _np.shape(b))
            if flops > FLOP_LIMIT:
                _limit(f"matrix product above {FLOP_LIMIT} multiply-adds")
            _check_size(size)
            return
        shape = _np.broadcast_shapes(_np.shape(a), _np.shape(b))
    except (ValueError, TypeError):
        return  # numpy reports incompatible operands itself
    _check_size(_product(shape), "a broadcast result")


def _check(op, a, b):
    if op == "*":
        for sequence, count in ((a, b), (b, a)):
            if isinstance(sequence, (str, bytes, bytearray, list, tuple)) and _is_int(count) \
                    and len(sequence) * count > ITEM_LIMIT:
                _limit(f"sequence repetition beyond {ITEM_LIMIT} items")
        if _is_int(a) and _is_int(b) and a.bit_length() + b.bit_length() > INT_BITS_LIMIT:
            _limit(f"integer beyond {INT_BITS_LIMIT} bits")
    elif op == "**":
        if _is_int(a) and _is_int(b) and b > 0 and abs(a) > 1 \
                and b * _math.log2(abs(a)) > INT_BITS_LIMIT:
            _limit(f"integer power beyond {INT_BITS_LIMIT} bits")
    elif op == "<<":
        if _is_int(a) and _is_int(b) and a and a.bit_length() + b > INT_BITS_LIMIT:
            _limit(f"integer shift beyond {INT_BITS_LIMIT} bits")
    _check_arrays(op, a, b)


_NUMBERS = frozenset({int, float, bool})
_INT_GROWTH = frozenset({"*", "**", "<<"})


def _binop(op, a, b):
    # Fast path for plain numbers, without nested calls: only integer growth is checked.
    if type(a) in _NUMBERS and type(b) in _NUMBERS:
        if op in _INT_GROWTH and type(a) is int and type(b) is int:
            if op == "*":
                grows = a.bit_length() + b.bit_length() > INT_BITS_LIMIT
            elif op == "**":
                grows = b > 0 and abs(a) > 1 and b * _math.log2(abs(a)) > INT_BITS_LIMIT
            else:
                grows = bool(a) and a.bit_length() + b > INT_BITS_LIMIT
            if grows:
                _limit(f"integer result beyond {INT_BITS_LIMIT} bits")
        return _OPERATORS[op](a, b)
    _check(op, a, b)
    return _OPERATORS[op](a, b)


def _checked(op, value, current=_UNKNOWN):
    """Guard an augmented assignment; the in-place operation itself stays Python's."""
    if current is not _UNKNOWN:
        if type(current) in _NUMBERS and type(value) in _NUMBERS and op not in _INT_GROWTH:
            return value
        _check(op, current, value)
    elif _is_int(value) and ((op == "*" and abs(value) > ITEM_LIMIT) or (op == "**" and value > 4096)
                             or (op == "<<" and value > INT_BITS_LIMIT)):
        _limit(f"augmented {op} with {value} is beyond the sandbox bounds")
    return value


def _compare(op, a, b):
    if not (type(a) in _NUMBERS and type(b) in _NUMBERS):
        _check_arrays(op, a, b)
    return _COMPARATORS[op](a, b)


def _range(*args):
    value = range(*args)
    try:
        count = len(value)
    except OverflowError:
        count = ITEM_LIMIT + 1
    if count > ITEM_LIMIT:
        _limit(f"range longer than {ITEM_LIMIT}")
    return value


def _pow(base, exp, mod=None):
    if mod is None:
        return _binop("**", base, exp)
    return pow(base, exp, mod)


def _iter(value):
    """One-argument iter only: iter(callable, sentinel) could loop forever in C."""
    return iter(value)


def _dir(value):
    return [name for name in builtins.dir(value) if not blocked_attribute(name)]


_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "callable", "chr", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "hash", "hex", "int", "isinstance", "len", "list", "map", "max",
    "min", "next", "oct", "ord", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip")
_EXCEPTIONS = (Exception, ArithmeticError, AssertionError, AttributeError, FloatingPointError,
               ImportError, IndexError, KeyError, LookupError, NameError, NotImplementedError,
               OverflowError, RecursionError, RuntimeError, StopIteration, TypeError,
               UnboundLocalError, ValueError, ZeroDivisionError)


# ----------------------------------------------------------------------------------------
# Module facades
# ----------------------------------------------------------------------------------------

def _module(name, members):
    module = types.ModuleType(name, "Sandbox facade of " + name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def _wrap(function, name, pre=None):
    """Expose a library callable as a plain function with size checks and no attributes."""
    def facade(*args, **kwargs):
        if pre is not None:
            pre(*args, **kwargs)
        result = function(*args, **kwargs)
        _check_result(result)
        return result
    facade.__name__ = facade.__qualname__ = name
    facade.__doc__ = getattr(function, "__doc__", None)
    return facade


def _check_result(result):
    if isinstance(result, _np.ndarray):
        _check_size(result.size)
        if result.dtype.kind == "O":
            _limit("object arrays are not available in code cells")
    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, _np.ndarray):
                _check_size(item.size)


def _shape_size(shape):
    if isinstance(shape, (list, tuple)):
        return _product(shape)
    return int(shape)


def _numeric_dtype(dtype):
    if dtype is not None and _np.dtype(dtype).kind == "O":
        _limit("object arrays are not available in code cells")


def _numpy_facade():
    np = _np
    array_like = (np.ndarray, list, tuple)

    def broadcast(*args, **kwargs):
        shapes = [np.shape(a) for a in args if isinstance(a, array_like)]
        if len(shapes) > 1:
            try:
                _check_size(_product(np.broadcast_shapes(*shapes)), "a broadcast result")
            except (ValueError, TypeError):
                pass

    def dtype_only(*args, dtype=None, **kwargs):
        _numeric_dtype(dtype)

    def shape_first(shape, *args, dtype=None, **kwargs):
        _numeric_dtype(dtype)
        _check_size(_shape_size(shape))

    def like(a, *args, dtype=None, shape=None, **kwargs):
        _numeric_dtype(dtype)
        _check_size(_shape_size(shape) if shape is not None else np.size(a))

    def eye(N, M=None, *args, **kwargs):
        _check_size(int(N) * int(N if M is None else M))

    def identity(n, *args, **kwargs):
        _check_size(int(n) * int(n))

    def arange(*args, **kwargs):
        bounds = [kwargs.get(k) for k in ("start", "stop", "step") if k in kwargs]
        values = list(args) + bounds
        if len(values) == 1:
            start, stop, step = 0, values[0], 1
        else:
            start, stop = values[0], values[1]
            step = values[2] if len(values) > 2 else 1
        if step:
            _check_size(max(0, _math.ceil((float(stop) - float(start)) / float(step))))

    def linspace(start, stop, num=50, *args, **kwargs):
        _check_size(int(num))

    def tile(A, reps):  # numpy's own parameter names, so keyword calls keep working
        reps = reps if isinstance(reps, (list, tuple)) else [reps]
        _check_size(np.size(A) * _product(reps))

    def repeat(a, repeats, axis=None):
        count = _product([repeats]) if _is_int(repeats) or isinstance(repeats, np.integer) \
            else int(np.sum(repeats))
        _check_size(np.size(a) * count if np.ndim(repeats) == 0 else count)

    def meshgrid(*xi, **kwargs):
        _check_size(_product(np.size(x) for x in xi) * max(1, len(xi)))

    def outer(a, b, *args, **kwargs):
        _check_size(np.size(a) * np.size(b))

    def contraction(cost):
        def check(a, b, *args, **kwargs):
            if isinstance(a, array_like) and isinstance(b, array_like):
                try:
                    flops, size = cost(np.shape(a), np.shape(b))
                except (ValueError, IndexError):
                    return  # numpy reports incompatible shapes itself
                if flops > FLOP_LIMIT:
                    _limit(f"product above {FLOP_LIMIT} multiply-adds")
                _check_size(size)
        return check

    def dot_cost(sa, sb):
        if len(sa) <= 2 and len(sb) <= 2:
            return _matmul_cost(sa, sb)
        size = _product(sa[:-1]) * _product(sb[:-2]) * sb[-1]
        return size * sa[-1], size

    def inner_cost(sa, sb):
        size = _product(sa[:-1]) * _product(sb[:-1])
        return size * (sa[-1] if sa else 1), size

    def diag(v, k=0):
        n = np.size(v)
        if np.ndim(v) == 1:
            _check_size((n + abs(int(k))) ** 2)

    def polyfit(x, y, deg, *args, **kwargs):
        if int(deg) > 64:
            _limit("polyfit degree above 64")

    def convolve(a, v, *args, **kwargs):
        if np.size(a) * np.size(v) > FLOP_LIMIT:
            _limit(f"convolution above {FLOP_LIMIT} multiply-adds")

    table = [
        (broadcast, "abs absolute sign sqrt square exp expm1 log log10 log2 log1p sin cos tan "
                    "arcsin arccos arctan arctan2 sinh cosh tanh hypot degrees radians deg2rad "
                    "rad2deg floor ceil round rint trunc mod fmod remainder minimum maximum fmin "
                    "fmax clip where isfinite isnan isinf isclose cross"),
        (dtype_only, "sum prod cumsum cumprod mean average median std var min max amin amax ptp "
                     "argmin argmax argsort sort unique any all nonzero argwhere count_nonzero "
                     "searchsorted digitize interp percentile quantile diff unwrap nanmean "
                     "nanmedian nanstd nanmin nanmax nansum cov corrcoef allclose array_equal trace "
                     "array asarray copy stack vstack hstack column_stack concatenate reshape ravel "
                     "squeeze expand_dims transpose swapaxes moveaxis flip fliplr flipud roll rot90 "
                     "atleast_1d atleast_2d take take_along_axis triu tril polyval split "
                     "array_split"),
        (shape_first, "zeros ones full"), (like, "zeros_like ones_like full_like"),
        (eye, "eye"), (identity, "identity"), (arange, "arange"), (linspace, "linspace"),
        (tile, "tile"), (repeat, "repeat"), (meshgrid, "meshgrid"), (outer, "outer"),
        (contraction(_matmul_cost), "matmul"), (contraction(dot_cost), "dot"),
        (contraction(inner_cost), "inner"), (diag, "diag"), (polyfit, "polyfit"),
        (convolve, "convolve")]
    members = {}
    for check, names in table:
        for name in names.split():
            if hasattr(np, name):
                members[name] = _wrap(getattr(np, name), name, check)

    def cubic(*args, **kwargs):
        for value in args:
            if isinstance(value, array_like) and np.size(value) > CUBIC_LIMIT:
                _limit(f"linear algebra on more than {CUBIC_LIMIT} elements")

    linalg = {}
    for name in ("norm", "inv", "pinv", "det", "solve", "lstsq", "eig", "eigh", "eigvals",
                 "eigvalsh", "svd", "qr", "cholesky", "matrix_rank", "slogdet"):
        if hasattr(np.linalg, name):
            linalg[name] = _wrap(getattr(np.linalg, name), name, None if name == "norm" else cubic)
    linalg["LinAlgError"] = np.linalg.LinAlgError
    members["linalg"] = _module("numpy.linalg", linalg)
    for name in ("pi", "e", "inf", "nan", "newaxis", "float64", "float32", "int64", "int32",
                 "uint8", "bool_"):
        members[name] = getattr(np, name)

    class _ArrayType(type):
        def __instancecheck__(cls, value):
            return isinstance(value, np.ndarray)

        def __subclasscheck__(cls, value):
            return isinstance(value, type) and issubclass(value, np.ndarray)

    class ndarray(metaclass=_ArrayType):
        """isinstance(value, np.ndarray) works; create arrays with np.array."""

        def __new__(cls, *args, **kwargs):
            raise SandboxLimitError("create arrays with np.array, np.zeros or np.ones")

    members["ndarray"] = ndarray
    return _module("numpy", members), members["linalg"]


def _math_facade():
    members = {name: getattr(_math, name) for name in dir(_math)
               if not name.startswith("_") and name not in ("factorial", "comb", "perm", "lcm", "prod")}

    def prod(iterable, *, start=1):
        result = start
        for value in iterable:
            result = _binop("*", result, value)
        return result

    members["prod"] = prod
    return _module("math", members)


def _count(n, r, kind):
    if r < 0:
        return 0
    if kind == "permutations":
        if r > n:
            return 0
        total = 1
        for i in range(r):
            total *= n - i
            if total > ITEM_LIMIT:
                break
        return total
    if kind == "cwr":
        n, r = n + r - 1, r
    if r > n:
        return 0
    r = min(r, n - r)
    total = 1
    for i in range(r):
        total = total * (n - i) // (i + 1)
        if total > ITEM_LIMIT:
            break
    return total


def _itertools_facade():
    it = _itertools

    def bounded(count, r=1):
        if count > ITEM_LIMIT or count * max(1, r) > 4 * ITEM_LIMIT:
            _limit(f"combinatoric iterator beyond {ITEM_LIMIT} items")

    def repeat(obj, times):
        if not _is_int(times) or not 0 <= times <= ITEM_LIMIT:
            _limit(f"repeat needs times within 0..{ITEM_LIMIT}")
        return it.repeat(obj, times)

    def product(*iterables, repeat=1):
        if not _is_int(repeat) or repeat < 0:
            raise ValueError("repeat must be a non-negative integer")
        pools = [tuple(pool) for pool in iterables]
        width = len(pools) * repeat
        if width > ITEM_LIMIT:
            bounded(ITEM_LIMIT + 1)
        total = 1
        for _ in range(repeat if pools else 0):
            for pool in pools:
                total *= len(pool)
                if total > ITEM_LIMIT:
                    bounded(total)
        bounded(total, width)
        return it.product(*pools, repeat=repeat)

    def permutations(iterable, r=None):
        pool = tuple(iterable)
        r = len(pool) if r is None else r
        bounded(_count(len(pool), r, "permutations"), r)
        return it.permutations(pool, r)

    def combinations(iterable, r):
        pool = tuple(iterable)
        bounded(_count(len(pool), r, "combinations"), r)
        return it.combinations(pool, r)

    def combinations_with_replacement(iterable, r):
        pool = tuple(iterable)
        bounded(_count(len(pool), r, "cwr"), r)
        return it.combinations_with_replacement(pool, r)

    def tee(iterable, n=2):
        if not _is_int(n) or not 0 <= n <= 1000:
            _limit("tee copies are limited to 1000")
        return it.tee(iterable, n)

    members = {name: getattr(it, name) for name in (
        "accumulate", "chain", "compress", "dropwhile", "filterfalse", "groupby", "islice",
        "pairwise", "starmap", "takewhile", "zip_longest") if hasattr(it, name)}
    members.update(repeat=repeat, product=product, permutations=permutations,
                   combinations=combinations,
                   combinations_with_replacement=combinations_with_replacement, tee=tee)
    return _module("itertools", members)


def _json_facade():
    def dumps(obj, *, indent=None, sort_keys=False, separators=None, ensure_ascii=True,
              allow_nan=True, default=None):
        if indent is not None and not ((_is_int(indent) and 0 <= indent <= 16)
                                       or (isinstance(indent, str) and len(indent) <= 16)):
            raise ValueError("indent must be at most 16")
        return _json.dumps(obj, indent=indent, sort_keys=sort_keys, separators=separators,
                           ensure_ascii=ensure_ascii, allow_nan=allow_nan, default=default)

    def loads(s):
        return _json.loads(s)

    return _module("json", {"dumps": dumps, "loads": loads, "JSONDecodeError": _json.JSONDecodeError})


def _functools_facade():
    names = ("reduce", "partial", "lru_cache", "cache", "cmp_to_key")
    return _module("functools", {n: getattr(_functools, n) for n in names if hasattr(_functools, n)})


def _statistics_facade():
    names = ("mean", "fmean", "geometric_mean", "harmonic_mean", "median", "median_low",
             "median_high", "median_grouped", "mode", "multimode", "pstdev", "pvariance", "stdev",
             "variance", "quantiles", "correlation", "covariance", "linear_regression",
             "NormalDist", "StatisticsError")
    return _module("statistics", {n: getattr(_statistics, n) for n in names if hasattr(_statistics, n)})


def _collections_facade():
    names = ("deque", "Counter", "OrderedDict", "defaultdict", "namedtuple", "ChainMap")
    return _module("collections", {n: getattr(_collections, n) for n in names})


def build_facades() -> dict:
    numpy, linalg = _numpy_facade()
    return {"math": _math_facade(), "numpy": numpy, "numpy.linalg": linalg,
            "json": _json_facade(), "itertools": _itertools_facade(),
            "functools": _functools_facade(), "statistics": _statistics_facade(),
            "collections": _collections_facade()}


# ----------------------------------------------------------------------------------------
# Plain data across the cell boundary
# ----------------------------------------------------------------------------------------

def to_plain(value, _depth=0, _budget=None):
    """Copy robo arguments into JSON-like data inside the cell thread; reject anything else.

    Conversion happens in the cell so that no agent object (generator, function) is ever
    iterated or called by the policy thread.
    """
    budget = [PLAIN_LIMIT] if _budget is None else _budget
    budget[0] -= 1
    if budget[0] < 0:
        raise RoboError(f"robo arguments are limited to {PLAIN_LIMIT} values")
    if _depth > 32:
        raise RoboError("robo arguments are nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (_np.ndarray, _np.generic)):
        if value.dtype.kind not in "biuf":
            raise RoboError("robo arguments accept numeric arrays only")
        if value.size > budget[0]:
            raise RoboError(f"robo arguments are limited to {PLAIN_LIMIT} values")
        budget[0] -= int(value.size)
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_plain(item, _depth + 1, budget) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RoboError("robo argument keys must be strings")
            result[key] = to_plain(item, _depth + 1, budget)
        return result
    raise RoboError("robo arguments must be plain data (numbers, strings, lists, dicts, "
                    f"numeric arrays), not {type(value).__name__}")


def summarize(value):
    """The final-expression value as plain data, or a truncated repr."""
    try:
        data = to_plain(value)
        if len(_json.dumps(data)) <= VALUE_LIMIT:
            return data
    except (RoboError, TypeError, ValueError):
        pass
    text = repr(value)
    return text if len(text) <= VALUE_LIMIT else text[:VALUE_LIMIT] + "..."


# ----------------------------------------------------------------------------------------
# The channel from a cell's robo object to the agent thread
# ----------------------------------------------------------------------------------------

class _Channel:
    __slots__ = ("_session",)

    def __init__(self, session):
        self._session = session

    def call(self, method, arguments):
        cell = self._session._current
        if cell is None or cell._ident != threading.get_ident():
            raise RoboError("robo is only available inside the running code cell")
        reply = cell._request(method, to_plain(arguments))
        if reply.kind == "error":
            raise RoboError(reply.value)
        if reply.kind == "failure":
            raise RoboFailure(reply.value["message"], reply.value["result"])
        return reply.value


# ----------------------------------------------------------------------------------------
# Cells and sessions
# ----------------------------------------------------------------------------------------

class Event:
    __slots__ = ("kind", "method", "arguments", "summary")

    def __init__(self, kind, method=None, arguments=None, summary=None):
        self.kind, self.method, self.arguments, self.summary = kind, method, arguments, summary


class Reply:
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    @classmethod
    def ok(cls, value):
        return cls("value", value)

    @classmethod
    def error(cls, message):
        return cls("error", str(message))

    @classmethod
    def failure(cls, message, result):
        return cls("failure", {"message": str(message), "result": result})

    @classmethod
    def stop(cls, kind):
        return cls("stop", kind)


class _Printer:
    def __init__(self):
        self._parts = []
        self._size = 0
        self._truncated = False

    def write(self, values, sep, end):
        if not isinstance(sep, str) or not isinstance(end, str):
            raise TypeError("print sep and end must be strings")
        text = sep.join(str(value) for value in values) + end
        room = STDOUT_LIMIT - self._size
        if len(text) > room:
            text, self._truncated = text[:max(0, room)], True
        if text:
            self._parts.append(text)
            self._size += len(text)

    def text(self):
        return "".join(self._parts)


_NO_VALUE = object()
_IMPORT_NAME = dis.opmap["IMPORT_NAME"]


class Cell:
    """One code cell in its own thread; the policy thread drives it like a coroutine."""

    def __init__(self, session, code, time_limit_s):
        self._session = session
        self._code = code
        self._events = queue.SimpleQueue()
        self._replies = queue.SimpleQueue()
        self._limit = float(time_limit_s)
        self._used = 0.0
        self._released = 0.0
        self._deadline = _math.inf
        self._abort = None
        self._critical = False
        self._ident = None
        self._printer = _Printer()
        self._value = _NO_VALUE
        self._thread = threading.Thread(target=self._main, name="r2si-code-cell", daemon=True)
        self.finished = False
        self.abandoned = False

    # -- policy thread ---------------------------------------------------------------
    def begin(self) -> Event:
        self._release()
        self._thread.start()
        return self._await()

    def resume(self, reply: Reply) -> Event:
        self._release()
        self._replies.put(reply)
        return self._await()

    def cancel(self):
        if self.finished:
            return
        self._abort = CellCancelled
        self._replies.put(Reply.stop(CellCancelled))
        self.finished = True

    def _release(self):
        self._released = time.perf_counter()
        self._deadline = self._released + max(0.0, self._limit - self._used)

    def _await(self) -> Event:
        try:
            event = self._events.get(timeout=max(0.0, self._limit - self._used) + STUCK_GRACE_S)
        except queue.Empty:
            # The cell holds the interpreter in one C-level call. Stop waiting; the thread
            # stops at its next bytecode and the next cell gets a fresh namespace.
            self._used = self._limit
            self._abort = CellTimeout
            self.abandoned = self.finished = True
            self._session._abandon(self)
            return Event("done", summary={
                "status": "timeout", "stdout": self._printer.text(),
                "stdout_truncated": self._printer._truncated, "value": None, "failure": None,
                "error": {"type": "CellTimeout", "line": None, "message":
                          "code cell did not return control after its compute limit; "
                          "the episode namespace was reset"},
                "namespace": [], "namespace_reset": True})
        self._used += time.perf_counter() - self._released
        if event.kind == "done":
            self.finished = True
        return event

    # -- cell thread -----------------------------------------------------------------
    def _trace(self, frame, event, arg):
        if not self._critical:
            if self._abort is not None:
                raise self._abort("code cell stopped")
            if time.perf_counter() > self._deadline:
                self._abort = CellTimeout
                raise CellTimeout(f"code cell exceeded {self._limit:g} s of Python compute")
        # Line events only for the agent's own frames; library and guard frames are still
        # checked at every call they make, without per-line overhead.
        return self._trace if frame.f_code.co_filename == "<cell>" else None

    def _request(self, method, arguments) -> Reply:
        self._critical = True
        try:
            self._events.put(Event("call", method=method, arguments=arguments))
            reply = self._replies.get()
        finally:
            self._critical = False
        if reply.kind == "stop":
            self._abort = reply.value
            raise reply.value("code cell stopped")
        return reply

    def _main(self):
        self._ident = threading.get_ident()
        sys.settrace(self._trace)
        try:
            try:
                exec(self._code, self._session._namespace)
                summary = self._summary("complete")
            except _CellStop as stop:
                summary = self._summary(stop.status, error=stop)
            except RoboFailure as failure:
                summary = self._summary("error", error=failure, failure=failure.result)
            except BaseException as error:  # agent errors end the cell, never the policy
                summary = self._summary("error", error=error)
        finally:
            sys.settrace(None)
        self._events.put(Event("done", summary=summary))

    def _summary(self, status, error=None, failure=None):
        result = {"status": status, "stdout": self._printer.text(),
                  "stdout_truncated": self._printer._truncated, "value": None,
                  "error": None, "failure": failure}
        if status == "complete" and self._value is not _NO_VALUE and self._value is not None:
            result["value"] = summarize(self._value)
        if error is not None:
            line = None
            for frame, number in traceback.walk_tb(error.__traceback__):
                if frame.f_code.co_filename == "<cell>":
                    line = number
            message = str(error)
            result["error"] = {"type": type(error).__name__, "line": line,
                               "message": message if len(message) <= 2000 else message[:2000] + "..."}
        result["namespace"] = self._session.user_names()
        return result


class Session:
    """The persistent namespace of one episode and its code cells."""

    PRELOADED = ("robo", "np", "math", "RoboError", "RoboFailure", "SandboxLimitError")

    def __init__(self, *, time_limit_s, robo_class):
        self._time_limit = float(time_limit_s)
        self._robo_class = robo_class
        self._channel = _Channel(self)
        self._current = None
        self._facades = build_facades()
        self._namespace = self._fresh()
        self.closed = False

    def _fresh(self):
        table = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}
        table.update({error.__name__: error for error in _EXCEPTIONS})
        table.update(range=_range, pow=_pow, iter=_iter, dir=_dir, print=self._print,
                     __import__=self._import, RoboError=RoboError, RoboFailure=RoboFailure,
                     SandboxLimitError=SandboxLimitError)
        return {"__builtins__": table, "__name__": "__cell__", "robo": self._robo_class(self._channel),
                "np": self._facades["numpy"], "math": self._facades["math"],
                "RoboError": RoboError, "RoboFailure": RoboFailure,
                "SandboxLimitError": SandboxLimitError, "__sbx_binop__": _binop,
                "__sbx_checked__": _checked, "__sbx_compare__": _compare,
                "__sbx_value__": self._record_value}

    def _print(self, *values, sep=" ", end="\n"):
        cell = self._current
        if cell is not None and cell._ident == threading.get_ident():
            cell._printer.write(values, sep, end)

    def _record_value(self, value):
        cell = self._current
        if cell is not None and cell._ident == threading.get_ident():
            cell._value = value

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        # The cell's own import statements get facades. Library C code that imports lazily
        # while a cell frame is current (e.g. numpy formatting a dtype) also reaches the
        # frame's builtins; it gets the real import. Agent code can never call this
        # function directly: its name is a dunder and validation accepts only facade imports.
        caller = sys._getframe(1)
        if caller.f_code.co_filename == "<cell>" and caller.f_code.co_code[caller.f_lasti] == _IMPORT_NAME:
            if level or name not in self._facades:
                raise ImportError(f"module {name!r} is not available in code cells")
            return self._facades[name] if fromlist else self._facades[name.partition(".")[0]]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _abandon(self, cell):
        if self._current is cell:
            self._current = None
        self._namespace = self._fresh()

    def user_names(self):
        names = [n for n in self._namespace if not n.startswith("__") and n not in self.PRELOADED]
        return sorted(names)[:64]

    def compile(self, code) -> Cell:
        """Validate a cell (SandboxViolation on any rule) and prepare it to run."""
        if self.closed:
            raise RuntimeError("The code session of this episode is closed")
        if self._current is not None and not self._current.finished:
            raise RuntimeError("A code cell is already running")
        cell = Cell(self, validate(code, self._facades), self._time_limit)
        self._current = cell
        return cell

    def close(self):
        """Episode reset: stop a suspended cell and drop the namespace."""
        self.closed = True
        cell, self._current = self._current, None
        if cell is not None and not cell.finished:
            cell.cancel()
            if cell._thread.is_alive():
                cell._thread.join(timeout=0.5)
        self._namespace = {}

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2023 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2021 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
"""

import ast
import builtins
import contextlib
import math
import re
import types
from math import isclose

# sympy related
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

# verl related
from verl.utils.py_functional import timeout_limit

# ---------------------------------------------------------------------------
# Security hardening (verl-project/verl#5331 - "Critical Security
# Vulnerabilities in Mathematical Evaluation and Serialization Code").
#
# `prediction` (and, to a lesser extent, `reference`) strings handled by this
# module originate from the policy model's own generated completion text,
# which is graded as part of the RL reward computation on *every* rollout.
# The original implementation called Python's builtin eval() directly on
# that text in a couple of places, which is a straightforward
# arbitrary-code-execution vector (e.g. a completion containing
# "__import__('os').system(...)" as part of its "answer").
#
# sympy.parsing.sympy_parser.parse_expr() is NOT a safe drop-in replacement
# for eval() by itself: its own docstring warns "this function uses eval,
# and thus shouldn't be used on unsanitized input". By default
# (global_dict=None) it even copies every Python builtin function --
# including __import__, eval, exec, open, compile, ... -- into the
# namespace it evaluates against, so
# `parse_expr("__import__('os').system(...)")` executes just as readily as
# a bare eval() call would (verified while writing this patch). The helpers
# below build a namespace that keeps sympy's own names (so numeric/symbolic
# parsing keeps working exactly as before) but omits the dangerous
# builtins, and separately validate the input before parsing.
#
# Note that removing dangerous builtins still cannot, by itself, stop
# "sandbox escape via live object-graph traversal" tricks such as
# `[].__class__.__base__.__subclasses__()` (walking the object graph this
# way needs no builtin function name at all). `handle_pi()` is fully immune
# to this because it is only ever handed a strictly-validated arithmetic
# expression (see `_safe_parse_expr`). For the general-purpose symbolic
# parsing elsewhere in this module we additionally reject any expression
# containing "__", which blocks the concrete gadget above (no legitimate
# LaTeX-derived math expression needs a double underscore); a fully
# rigorous fix for arbitrary untrusted symbolic input would require running
# the parser in a real sandbox (subprocess / restricted syscalls) rather
# than namespace or string filtering.
# ---------------------------------------------------------------------------

_DANGEROUS_BUILTIN_NAMES = frozenset(
    {
        "__import__",
        "__build_class__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "help",
        "exit",
        "quit",
        "memoryview",
    }
)

_SAFE_ARITHMETIC_EXPR_RE = re.compile(r"^[\d\s+\-*/().eE]+$")


def _safe_sympy_global_dict() -> dict:
    """Namespace for ``parse_expr`` that mirrors its own default namespace
    (``global_dict=None``), except that it omits builtins with no
    legitimate use in a math expression that are the standard vector for
    turning an expression parser into arbitrary code execution.
    """
    global_dict: dict = {}
    exec("from sympy import *", global_dict)  # trusted, fixed source string
    for name, obj in vars(builtins).items():
        if name in _DANGEROUS_BUILTIN_NAMES:
            continue
        if isinstance(obj, types.BuiltinFunctionType):
            global_dict[name] = obj
    global_dict["max"] = global_dict["Max"]
    global_dict["min"] = global_dict["Min"]
    global_dict["__builtins__"] = {}
    return global_dict


def _safe_parse_expr(string: str):
    """Safely evaluate the plain arithmetic expression built by
    ``handle_pi`` (e.g. ``"2*3.141592653589793"``), replacing the previous
    ``eval(string)`` call (verl-project/verl#5331). Only a strict allow-list
    of arithmetic characters is accepted before parsing, which -- unlike a
    bare/naive ``parse_expr()`` call -- also closes the object-graph
    traversal gadgets described above.
    """
    if not _SAFE_ARITHMETIC_EXPR_RE.match(string):
        raise ValueError(f"refusing to evaluate non-arithmetic expression: {string!r}")
    return parse_expr(string, global_dict=_safe_sympy_global_dict(), evaluate=True)


def _safe_symbolic_parse_expr(expr: str):
    """Parse a general (possibly symbolic) expression through the hardened
    namespace above (verl-project/verl#5331). Also rejects any expression
    containing a double underscore, which blocks Python dunder-attribute
    gadgets such as ``"[].__class__.__base__.__subclasses__()"`` that
    survive even a builtins-free namespace. This is a pragmatic,
    defense-in-depth filter -- not a substitute for real sandboxing -- since
    no legitimate LaTeX-derived math expression needs a double underscore.
    """
    if "__" in expr:
        raise ValueError(f"refusing to evaluate expression containing '__': {expr!r}")
    return parse_expr(expr, global_dict=_safe_sympy_global_dict())


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer, pi) -> str:
    # checking if answer is $<number> and removing $ in that case to compare
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    # handle base
    answer = handle_base(answer)

    # handle pi
    answer = handle_pi(answer, pi)

    return answer


def handle_base(x) -> str:
    if isinstance(x, str) and "_" in x:
        # Due to base
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x


def handle_pi(string, pi):
    if isinstance(string, str) and "\\pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("\\pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]

            # Find the next occurrence of "\pi"
            idx = string.find("\\pi", idx + 1)

        # Evaluate the arithmetic expression. SECURITY (verl-project/verl#5331):
        # this used to call eval(string) directly on the model's own answer
        # text, which is arbitrary code execution. _safe_parse_expr() only
        # accepts a strict arithmetic-character allow-list and evaluates it
        # through a builtins-free sympy namespace; we only keep the result
        # when it collapses to a plain number, matching what eval() produced
        # for the numeric "N*math.pi"-style expressions this function
        # builds. Any other outcome (validation failing, parsing raising, or
        # the result not being numeric) leaves `string` as the
        # substituted-but-unevaluated text, exactly as eval() raising an
        # exception did before.
        with contextlib.suppress(Exception):
            parsed = _safe_parse_expr(string)
            if parsed.is_number:
                string = float(parsed)

    return string


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            # number questions
            gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=True)
            ]
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            return bool(
                all(
                    [
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                        for i in range(len(pred_parts))
                    ]
                )
            )

    # if we have point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=False)
            ]
        ):
            return True

    # if reference is a matrix
    if r"\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            # SECURITY (verl-project/verl#5331): route through the hardened,
            # builtins-restricted namespace instead of parse_expr()'s own
            # (unsafe-by-default) namespace -- see module-level comment above
            # _safe_sympy_global_dict for why this matters.
            pred_matrix = _safe_symbolic_parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                [
                    math_equal(pred, ref, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
                ]
            ):
                return True
        except Exception:
            pass
    elif r"\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        # SECURITY (verl-project/verl#5331): eval(prediction) executed the
        # model's own answer text as Python code. ast.literal_eval() only
        # accepts Python literal syntax (lists/tuples/dicts/numbers/strings/
        # booleans/None) -- it parses "[[1, 2], [3, 4]]" exactly like eval()
        # did, but it is structurally incapable of calling functions,
        # importing modules, or evaluating attribute access, so it cannot
        # execute code.
        if isinstance(ast.literal_eval(prediction), list):
            try:
                pred_matrix = ast.literal_eval(prediction)
                # ref_matrix_items = reference.split()[1:-1:2]
                ref_matrix_items = (
                    reference.removeprefix(r"\\begin{pmatrix}")
                    .removeprefix(r"\begin{pmatrix}")
                    .removesuffix(r"\\end{pmatrix}")
                    .removesuffix(r"\end{pmatrix}")
                )
                ref_matrix_items = ref_matrix_items.split("\\")
                ref_matrix_items = [row.split("&") if "&" in row else row for row in ref_matrix_items]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    [
                        math_equal(pred, ref, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
                    ]
                ):
                    return True
            except Exception:
                pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    def _parse(s):
        # SECURITY (verl-project/verl#5331): route parse_expr through the
        # hardened namespace (see module-level comment above
        # _safe_sympy_global_dict) instead of its own unsafe-by-default one.
        for f in [_safe_symbolic_parse_expr, parse_latex]:
            try:
                with timeout_limit(seconds=timeout):
                    return f(s)
            except TimeoutError:
                print(f"Parsing timed out for {s}")
                continue
            except Exception:
                continue
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with timeout_limit(seconds=timeout):
            if simplify(a - b) == 0:
                return True
    except TimeoutError:
        print(f"Simplification timed out for {a} - {b}")
        pass
    except Exception:
        pass

    try:
        with timeout_limit(seconds=timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except TimeoutError:
        print(f"Numerical evaluation timed out for {a}, {b}")
        pass
    except Exception:
        pass
    return False


def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Intarval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                return f"({inner_content})"

    return prediction

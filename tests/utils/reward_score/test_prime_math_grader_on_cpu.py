# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Security regression tests for verl/utils/reward_score/prime_math/grader.py
(verl-project/verl#5331 - "Critical Security Vulnerabilities in Mathematical
Evaluation and Serialization Code").

`grader.py` grades the policy model's own generated completion text as part
of the RL reward computation, so it must never execute that text as code.
The original implementation called Python's builtin ``eval()`` directly on
model output in a few places. These tests:

  1. Prove legitimate inputs still produce the same results as before the
     fix (regression coverage for `handle_pi` and the matrix-literal path).
  2. Prove concrete adversarial payloads that used to achieve real code
     execution (verified against the pre-fix implementation while writing
     this patch -- each PoC below created a real file on disk) no longer do
     anything beyond failing to match / raising a caught exception.
"""

import math
import os
import tempfile

from verl.utils.reward_score.prime_math.grader import (
    _safe_symbolic_parse_expr,
    handle_pi,
    math_equal,
    symbolic_equal,
)

# ---------------------------------------------------------------------------
# Legitimate-input regression cases
# ---------------------------------------------------------------------------


def test_handle_pi_legit_multiple():
    # "N\pi" -> N * pi, same as the previous eval()-based implementation.
    result = handle_pi("2\\pi", math.pi)
    assert isinstance(result, float)
    assert math.isclose(result, 2 * math.pi, rel_tol=1e-9)


def test_handle_pi_legit_bare():
    # "\pi" with no preceding digit -> 1 * pi.
    result = handle_pi("\\pi", math.pi)
    assert isinstance(result, float)
    assert math.isclose(result, math.pi, rel_tol=1e-9)


def test_handle_pi_legit_decimal_multiple():
    result = handle_pi("3.5\\pi", math.pi)
    assert isinstance(result, float)
    assert math.isclose(result, 3.5 * math.pi, rel_tol=1e-9)


def test_handle_pi_no_pi_untouched():
    # No "\pi" substring -> the function must not touch the string at all.
    assert handle_pi("hello world", math.pi) == "hello world"


def test_handle_pi_non_numeric_leaves_string_substituted_unevaluated():
    # Mirrors the pre-fix eval() semantics: when the expression cannot be
    # reduced to a plain number, the *substituted* (but unevaluated) string
    # is returned unchanged, exactly as when eval() used to raise and get
    # suppressed by contextlib.suppress(Exception).
    result = handle_pi("x\\pi", math.pi)
    assert result == f"1*{math.pi}x" or result == f"x1*{math.pi}"
    # (the exact placement mirrors the substitution logic; the key invariant
    # is that it is NOT reduced to a float since "x" is not a plain number)
    assert not isinstance(result, float)


def test_matrix_bracket_literal_matches_pmatrix_reference():
    # "[5, 6]" vs a 2-row \begin{pmatrix} reference: this exercises the
    # ast.literal_eval() path that replaced eval(prediction) in math_equal's
    # bracket-matrix branch. Verified to also return True against the
    # pre-fix eval()-based implementation, i.e. identical output.
    prediction = "[5, 6]"
    reference = "\\begin{pmatrix} 5 \\ 6 \\end{pmatrix}"
    assert math_equal(prediction, reference) is True


def test_matrix_matrix_prefixed_literal_matches_pmatrix_reference():
    prediction = "Matrix([[1, 2]])"
    reference = r"\begin{pmatrix} 1 & 2 \end{pmatrix}"
    assert math_equal(prediction, reference) is True


# ---------------------------------------------------------------------------
# Adversarial-input security cases
# ---------------------------------------------------------------------------


def test_handle_pi_blocks_code_execution():
    """
    Before this fix, handle_pi() called eval() on the (\\pi-substituted)
    answer text. This payload is a real, verified PoC: against the
    pre-fix implementation it executed os.system(...) and created a file
    on disk. After the fix it must not execute anything.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = os.path.join(tmpdir, "pwned")
        payload = f"__import__('os').system('touch {marker}') and 1\\pi"

        # Must not raise out of handle_pi (contextlib.suppress covers it,
        # same as it always did for eval() failures) and must not execute.
        result = handle_pi(payload, math.pi)

        assert not os.path.exists(marker), "adversarial payload executed code via handle_pi()"
        # No crash, and no numeric coercion of attacker-controlled code.
        assert not isinstance(result, float)


def test_handle_pi_blocks_subclasses_gadget():
    """
    Even without any dangerous builtin name, walking the live object graph
    (e.g. to reach subprocess.Popen) is a classic "safe eval" bypass. This
    must not raise an unhandled exception nor return a live object graph.
    """
    payload = "[].__class__.__base__.__subclasses__()\\pi"
    result = handle_pi(payload, math.pi)
    # The strict arithmetic allow-list in _safe_parse_expr rejects this
    # outright, so `string` is left as the substituted-but-unevaluated text.
    assert not isinstance(result, float)
    assert not isinstance(result, list)


def test_matrix_bracket_branch_blocks_code_execution():
    """
    Before this fix, math_equal()'s bracket-matrix branch called
    eval(prediction) directly. This payload is a real, verified PoC:
    against the pre-fix implementation it opened and wrote a file on disk
    and the overall math_equal() call returned False *silently* (masking
    the fact that arbitrary code had just executed). After the fix it must
    not execute anything, whether or not an exception is raised.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = os.path.join(tmpdir, "pwnedmatrix")
        payload = f"[open('{marker}','w').write('x')]"
        reference = r"\begin{pmatrix} 5 \end{pmatrix}"

        try:
            math_equal(payload, reference)
        except Exception:
            # ast.literal_eval() raising ValueError/SyntaxError for
            # non-literal syntax is acceptable ("graceful ... exception,
            # not crash the process or execute code") -- the pre-existing
            # code around this branch is not wrapped in try/except either.
            pass

        assert not os.path.exists(marker), "adversarial payload executed code via the bracket-matrix branch"


def test_matrix_matrix_branch_blocks_code_execution():
    """
    Before this fix, math_equal()'s "Matrix(...)"-prefixed branch called
    (an unrestricted) parse_expr(prediction) -- not a literal eval() call,
    but sympy documents that parse_expr() "uses eval, and thus shouldn't be
    used on unsanitized input". This payload is a real, verified PoC:
    against the pre-fix implementation it wrote a file on disk AND
    math_equal() returned True (a full reward for an exploit). After the
    fix it must not execute anything.
    """
    # Deliberately not tempfile.TemporaryDirectory(): its random component can
    # contain "_", which would trip the unrelated handle_base() "look for a
    # numeric base suffix" behavior before the payload ever reaches the
    # "Matrix(...)" branch this test targets. Use a fixed, underscore-free
    # marker path instead.
    marker = os.path.join(tempfile.gettempdir(), "verl-issue-5331-matrix-fn-marker")
    if os.path.exists(marker):
        os.remove(marker)
    try:
        payload = f"Matrix(open('{marker}','w').write('x') and [[1,2]] or [[1,2]])"
        reference = r"\begin{pmatrix} 1 & 2 \end{pmatrix}"

        result = math_equal(payload, reference)

        assert not os.path.exists(marker), "adversarial payload executed code via the 'Matrix(...)' branch"
        assert result is False
    finally:
        if os.path.exists(marker):
            os.remove(marker)


def test_safe_symbolic_parse_expr_blocks_subclasses_gadget():
    """
    Direct unit test of the hardened symbolic-expression helper used by
    both the "Matrix(...)" branch and symbolic_equal(): the classic
    sandbox-escape gadget (which needs no dangerous builtin name at all,
    only live object-graph traversal) must be rejected outright.
    """
    payload = "[].__class__.__base__.__subclasses__()"
    try:
        result = _safe_symbolic_parse_expr(payload)
    except Exception:
        return  # rejecting via exception is the expected, safe outcome
    # If it didn't raise, it must not have returned the live class list
    # (i.e. it must not have actually walked the object graph).
    assert not isinstance(result, list)


def test_symbolic_equal_blocks_code_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = os.path.join(tmpdir, "pwnedsymbolic")
        payload = f"__import__('os').system('touch {marker}')"

        result = symbolic_equal(payload, "1", 1e-4, timeout=5)

        assert not os.path.exists(marker), "adversarial payload executed code via symbolic_equal()"
        assert result is False

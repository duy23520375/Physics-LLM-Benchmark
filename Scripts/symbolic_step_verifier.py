"""Symbolic step-verifier prototype (Stage 2 augmentation).

Motivation: the Qwen2.5-7B diagnostic tutor localizes a student's first error by
comparing against a gold solution in natural language. Two failure modes recur:

  (1) surface-diff: it flags a step that is written differently but is
      mathematically EQUIVALENT to the gold (e.g. Delta_t = 2*pi/(omega*2)
      is identical to pi/omega), mislocalizing the error;
  (2) confirmation bias: when two compensating errors cancel so the final
      answer matches the gold anchor, the tutor accepts a WRONG derivation
      as correct (the double-cancellation case).

A symbolic step-verifier removes both. Each student assertion `lhs == rhs` is
checked with SymPy `simplify(lhs - rhs) == 0` (value-equivalence, not string
match), optionally under the governing physical identities and numeric
substitutions. The FIRST assertion that is not an identity is the true first
error -- including wrong intermediate steps that endpoint-matching cannot see.

Run:  python scripts/symbolic_step_verifier.py
"""
import sympy as sp

omega, A, v_max, a_max, T, t = sp.symbols('omega A v_max a_max T t', positive=True)
pi = sp.pi
_SYM = dict(omega=omega, A=A, v_max=v_max, a_max=a_max, T=T, t=t, pi=pi, sqrt=sp.sqrt)


def verify_chain(assertions, relations=None, subs=None, tol=1e-9):
    """Check an ordered chain of student equalities.

    assertions : list of (label, lhs, rhs); the student asserts lhs == rhs.
    relations  : dict {symbol: expr} of governing physical identities to substitute
                 (e.g. {'a_max': 'omega**2*A'}).
    subs       : dict of numeric substitutions for a numeric fallback check.
    Returns (first_bad_label_or_None, trace) where trace is [(label, 'OK'|'ERROR')].
    """
    rel = {sp.sympify(k, locals=_SYM): sp.sympify(v, locals=_SYM)
           for k, v in (relations or {}).items()}
    nsub = {sp.sympify(k, locals=_SYM): (sp.nsimplify(v) if isinstance(v, str) else v)
            for k, v in (subs or {}).items()}
    trace, first = [], None
    for label, lhs, rhs in assertions:
        L = sp.sympify(lhs, locals=_SYM).subs(rel)
        R = sp.sympify(rhs, locals=_SYM).subs(rel)
        ok = (sp.simplify(L - R) == 0)
        if not ok and nsub:                      # numeric identity fallback
            try:
                ok = abs(complex(sp.N((L - R).subs(nsub)))) < tol
            except Exception:
                pass
        trace.append((label, 'OK' if ok else 'ERROR'))
        if not ok and first is None:
            first = label
    return first, trace


# --- The three marquee cases, encoded from the real student submissions. ---
CASES = {
    # student: Delta_t = T/2 = 2*pi/(omega*2) = pi/(4*pi) = 0.125  (gold: 0.25)
    "sub_01": dict(
        assertions=[
            ("T/2 = 2*pi/(omega*2)",       "T/2",            "2*pi/(omega*2)"),
            ("2*pi/(omega*2) = pi/(4*pi)", "2*pi/(omega*2)", "pi/(4*pi)"),
            ("pi/(4*pi) = 0.125",          "pi/(4*pi)",      "0.125"),
        ],
        relations={'T': '2*pi/omega'}, subs={'omega': '4*pi'},
    ),
    # student: a_max=(v_max/omega)^2 ; omega^2=a_max/v_max ; A=v_max/omega=4
    # two compensating errors -> right final answer (double cancellation)
    "sub_02": dict(
        assertions=[
            ("a_max = (v_max/omega)**2", "a_max",    "(v_max/omega)**2"),
            ("omega**2 = a_max/v_max",   "omega**2", "a_max/v_max"),
            ("A = v_max/omega",          "v_max/omega", "A"),
        ],
        relations={'a_max': 'omega**2*A', 'v_max': 'omega*A'},
    ),
    # student: v_tb = A*sqrt(3)/(T/3) = 9*A/T   (correct: 3*sqrt(3)*A/T)
    "sub_04": dict(
        assertions=[
            ("A*sqrt(3)/(T/3) = 3*A*sqrt(3)/T (gold)", "A*sqrt(3)/(T/3)", "3*A*sqrt(3)/T"),
            ("A*sqrt(3)/(T/3) = 9*A/T (student)",      "A*sqrt(3)/(T/3)", "9*A/T"),
        ],
    ),
}


def main():
    for name, spec in CASES.items():
        first, trace = verify_chain(**spec)
        print(f"\n=== {name} ===  first symbolic error: {first}")
        for label, status in trace:
            print(f"   [{status:5}] {label}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# A1.7 target 1 (README.md). Every stage resumes from its checkpoint, so it is safe to start again.
# Queue job (tools/queue): the worker signals only this script, so it passes SIGTERM on to the
# engine and waits for its checkpoint. A17_UCMD: engine with spec 0.1.2 pair overrides (b9f161e).
# A17_PYTHON: a Python with numpy and h5py (default: the ucmd-analysis environment next door).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
UCMD=${A17_UCMD:-$HOME/.cache/ucmd-a17/target/release/ucmd}
PYTHON=${A17_PYTHON:-../../../ucmd-analysis/.venv/bin/python}
export PYTHONPATH=../../src
mkdir -p out
child=
stopped=
trap 'stopped=1; [ -n "$child" ] && kill -TERM "$child" 2> /dev/null' TERM INT

stage() { # NAME: ucmd run NAME.toml, from its checkpoint if there is one
    [ -e "out/$1/done" ] && return 0
    local args=(run "$1.toml")
    [ -e "out/$1/checkpoint.ucchk" ] && args+=(--restart "out/$1/checkpoint.ucchk")
    "$UCMD" "${args[@]}" < /dev/null &
    child=$!
    wait "$child"
    local code=$?
    # a signal interrupts `wait`; go on waiting until the engine has really ended
    while kill -0 "$child" 2> /dev/null; do
        wait "$child"
        code=$?
    done
    child=
    [ -n "$stopped" ] && exit 143
    [ "$code" -eq 0 ] || exit "$code"
    touch "out/$1/done"
}

[ -e out/start.ucsys/system.json ] || "$PYTHON" make_system.py out/start.ucsys || exit 1
stage equilibrate
[ -e out/equilibrated.ucsys/system.json ] ||
    "$UCMD" export equilibrate.toml --checkpoint out/equilibrate/checkpoint.ucchk \
        --out out/equilibrated.ucsys || exit 1
stage npt
[ -e out/npt-end.ucsys/system.json ] ||
    "$UCMD" export npt.toml --checkpoint out/npt/checkpoint.ucchk --out out/npt-end.ucsys || exit 1
[ -e out/nvt-start.ucsys/system.json ] ||
    "$PYTHON" set_volume.py out/npt-end.ucsys out/npt/thermo.tsv out/nvt-start.ucsys || exit 1
stage nvt
"$PYTHON" analyze.py out > out/analyze.log || exit 1

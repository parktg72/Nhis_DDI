#!/usr/bin/env bash
set -u
set -o pipefail

if ! command -v jq >/dev/null 2>&1; then
    printf '%s\n' '{"status":"error","error":"dependency_missing","dependency":"jq"}'
    exit 3
fi

json_error() {
    local error_name=$1
    local detail_name=${2:-}
    local detail_value=${3:-}

    if [[ -n "$detail_name" ]]; then
        jq -n \
            --arg error "$error_name" \
            --arg detail_name "$detail_name" \
            --arg detail_value "$detail_value" \
            '{status: "error", error: $error} + {($detail_name): $detail_value}'
    else
        jq -n --arg error "$error_name" \
            '{status: "error", error: $error}'
    fi
}

validate_bounded_decimal() {
    local candidate=$1
    local maximum=$2
    local normalized=$candidate

    [[ $candidate =~ ^[0-9]+$ ]] || return 1
    while [[ ${#normalized} -gt 1 && $normalized == 0* ]]; do
        normalized=${normalized#0}
    done
    [[ $normalized != 0 ]] || return 1
    if (( ${#normalized} > ${#maximum} )) \
        || { (( ${#normalized} == ${#maximum} )) \
            && [[ $normalized > $maximum ]]; }; then
        return 1
    fi

    validated_decimal=$normalized
}

if (( $# < 2 )) || [[ -z ${2:-} ]]; then
    json_error "brief_missing"
    exit 2
fi

if (( $# != 2 )); then
    json_error "invalid_arguments"
    exit 2
fi

provider=$1
brief_arg=$2

case "$provider" in
    claude|claude-advisor|agy)
        ;;
    *)
        json_error "provider_not_allowed"
        exit 2
        ;;
esac

for dependency in timeout realpath sed; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        json_error "dependency_missing" "dependency" "$dependency"
        exit 3
    fi
done

script_path=$(realpath -e -- "${BASH_SOURCE[0]}")
script_dir=${script_path%/*}
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)

if ! brief_path=$(realpath -e -- "$brief_arg" 2>/dev/null) \
    || [[ ! -f "$brief_path" ]]; then
    json_error "brief_not_found"
    exit 2
fi

case "$brief_path" in
    "$repo_root"/*)
        ;;
    *)
        json_error "brief_outside_repo"
        exit 2
        ;;
esac

case "$provider" in
    claude|claude-advisor)
        cli=claude
        ;;
    agy)
        cli=agy
        ;;
esac

if ! command -v "$cli" >/dev/null 2>&1; then
    json_error "cli_not_found" "cli" "$cli"
    exit 3
fi

timeout_candidate=${AGENT_ADAPTER_TIMEOUT:-300}
if ! validate_bounded_decimal "$timeout_candidate" 3600; then
    json_error "invalid_timeout"
    exit 2
fi
timeout_seconds=$validated_decimal

kill_after_candidate=${AGENT_ADAPTER_KILL_AFTER:-5}
if ! validate_bounded_decimal "$kill_after_candidate" 3600; then
    json_error "invalid_kill_after"
    exit 2
fi
kill_after_seconds=$validated_decimal

work_dir=
if ! work_dir=$(mktemp -d "${TMPDIR:-/tmp}/agent-adapter.XXXXXX" 2>/dev/null) \
    || [[ -z "$work_dir" || ! -d "$work_dir" || "$work_dir" == / ]]; then
    if [[ -n "$work_dir" && "$work_dir" != / && -d "$work_dir" ]]; then
        rmdir -- "$work_dir" 2>/dev/null || :
    fi
    json_error "tempdir_failed"
    exit 3
fi

uncanonical_work_dir=$work_dir
if ! work_dir=$(realpath -e -- "$uncanonical_work_dir" 2>/dev/null) \
    || [[ -z "$work_dir" || ! -d "$work_dir" || "$work_dir" == / ]]; then
    if [[ ${work_dir:-} != / ]]; then
        rmdir -- "$uncanonical_work_dir" 2>/dev/null || :
    fi
    json_error "tempdir_failed"
    exit 3
fi

stdout_file="$work_dir/stdout"
stderr_file="$work_dir/stderr"
sanitized_file="$work_dir/stderr.sanitized"
timeout_stderr_file="$work_dir/timeout.stderr"

cleanup() {
    rm -f -- \
        "$stdout_file" "$stderr_file" "$sanitized_file" "$timeout_stderr_file" \
        2>/dev/null
    rmdir -- "$work_dir" 2>/dev/null || :
}

handle_signal() {
    local signal_exit_code=$1
    trap - EXIT HUP INT TERM
    cleanup
    exit "$signal_exit_code"
}

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if ! (
    : >"$stdout_file"
    : >"$stderr_file"
    : >"$sanitized_file"
    : >"$timeout_stderr_file"
) 2>/dev/null \
    || [[ ! -f "$stdout_file" \
        || ! -f "$stderr_file" \
        || ! -f "$sanitized_file" \
        || ! -f "$timeout_stderr_file" ]]; then
    json_error "capture_setup_failed"
    exit 3
fi

brief_content=$(<"$brief_path")

case "$provider" in
    claude)
        command=(
            claude -p "$brief_content"
            --permission-mode plan
            --tools Read,Grep,Glob
            --output-format json
            --no-session-persistence
        )
        ;;
    claude-advisor)
        advisor_prompt="Call the built-in advisor exactly once before any other tool. Do not call it again. Use the advisor feedback to answer this read-only brief, then return the required evidence fields.

$brief_content"
        command=(
            claude -p "$advisor_prompt"
            --permission-mode plan
            --tools advisor
            --settings "$repo_root/.claude/settings.json"
            --output-format stream-json
            --verbose
            --no-session-persistence
        )
        ;;
    agy)
        command=(
            agy
            --sandbox
            --mode plan
            --add-dir "$repo_root"
            --print "$brief_content"
            --print-timeout "${timeout_seconds}s"
        )
        ;;
esac

start_seconds=$SECONDS
(
    cd -- "$repo_root" || exit 3
    LC_ALL=C timeout --verbose --signal=TERM \
        --kill-after="${kill_after_seconds}s" "${timeout_seconds}s" \
        bash -c 'provider_stderr=$1; shift; exec "$@" 2>"$provider_stderr"' \
        adapter-provider "$stderr_file" "${command[@]}"
) >"$stdout_file" 2>"$timeout_stderr_file"
exit_code=$?
duration_seconds=$((SECONDS - start_seconds))

timeout_sent_term=false
timeout_sent_kill=false
while IFS= read -r diagnostic_line || [[ -n "$diagnostic_line" ]]; do
    if [[ $diagnostic_line == "timeout: sending signal TERM to command 'bash'" ]]; then
        timeout_sent_term=true
    elif [[ $diagnostic_line == "timeout: sending signal KILL to command 'bash'" ]]; then
        timeout_sent_kill=true
    fi
done <"$timeout_stderr_file"

case "$exit_code" in
    0)
        status=ok
        ;;
    124)
        if [[ $timeout_sent_term == true ]]; then
            status=timeout
        else
            status=error
        fi
        ;;
    137)
        if [[ $timeout_sent_kill == true ]]; then
            status=timeout
        else
            status=error
        fi
        ;;
    *)
        status=error
        ;;
esac

if ! {
    while IFS= read -r diagnostic_line || [[ -n "$diagnostic_line" ]]; do
        printf '%s\n' "$diagnostic_line"
    done <"$timeout_stderr_file"
} >>"$stderr_file" 2>/dev/null; then
    json_error "capture_setup_failed"
    exit 3
fi

if ! LC_ALL=C sed -E \
    -e 's/(Authorization[[:space:]]*:[[:space:]]*Bearer[[:space:]]+)[^[:space:]\",}]+/\1[REDACTED]/gI' \
    -e 's/(\"?(AWS_SECRET_ACCESS_KEY|API_KEY|TOKEN|SECRET|PASSWORD)\"?[[:space:]]*[=:][[:space:]]*\"?)[^\"[:space:],}]+/\1[REDACTED]/gI' \
    -e 's/sk-[A-Za-z0-9_-]+/[REDACTED]/gI' \
    "$stderr_file" >"$sanitized_file" 2>/dev/null; then
    json_error "redaction_failed"
    exit 3
fi

jq -n \
    --arg status "$status" \
    --arg provider "$provider" \
    --arg brief "$brief_path" \
    --argjson exit_code "$exit_code" \
    --argjson duration_seconds "$duration_seconds" \
    --rawfile stdout "$stdout_file" \
    --rawfile stderr_sanitized "$sanitized_file" \
    '{
        status: $status,
        provider: $provider,
        brief: $brief,
        exit_code: $exit_code,
        duration_seconds: $duration_seconds,
        stdout: $stdout,
        stderr_sanitized: $stderr_sanitized
    }'

exit "$exit_code"

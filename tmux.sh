#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
사용법:
  bash tmux.sh            # 세션 목록 표시 후 숫자 선택 attach
  bash tmux.sh <세션이름>  # 해당 세션으로 바로 attach
EOF
}

get_sessions() {
  tmux list-sessions -F "#{session_name}" 2>/dev/null || true
}

attach_session() {
  local session_name="$1"
  if ! tmux has-session -t "$session_name" 2>/dev/null; then
    echo "오류: tmux 세션 '$session_name'이(가) 존재하지 않습니다."
    exit 1
  fi
  tmux attach-session -t "$session_name"
}

read_number_key() {
  local key="" next="" seq=""
  while true; do
    read -rsn1 key || return 1

    # 일반 숫자키
    case "$key" in
    [0-9])
      printf "%s" "$key"
      return 0
      ;;
    q | Q)
      printf "q"
      return 0
      ;;
    "") continue ;;
    esac

    # ESC 시퀀스 처리(일부 키패드: ESC O p..y)
    if [[ "$key" == $'\e' ]]; then
      read -rsn1 -t 0.05 next || { continue; }
      if [[ "$next" == "O" ]]; then
        read -rsn1 -t 0.05 seq || continue
        case "$seq" in
        p)
          printf "0"
          return 0
          ;;
        q)
          printf "1"
          return 0
          ;;
        r)
          printf "2"
          return 0
          ;;
        s)
          printf "3"
          return 0
          ;;
        t)
          printf "4"
          return 0
          ;;
        u)
          printf "5"
          return 0
          ;;
        v)
          printf "6"
          return 0
          ;;
        w)
          printf "7"
          return 0
          ;;
        x)
          printf "8"
          return 0
          ;;
        y)
          printf "9"
          return 0
          ;;
        esac
      fi
    fi
  done
}

main() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "오류: tmux가 설치되어 있지 않습니다."
    exit 1
  fi

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
  fi

  if [[ $# -gt 1 ]]; then
    print_usage
    exit 1
  fi

  if [[ $# -eq 1 ]]; then
    attach_session "$1"
    exit 0
  fi

  mapfile -t sessions < <(get_sessions)
  if [[ ${#sessions[@]} -eq 0 ]]; then
    echo "오류: 현재 실행 중인 tmux 세션이 없습니다."
    exit 1
  fi

  echo "tmux 세션 목록:"
  for i in "${!sessions[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${sessions[$i]}"
  done
  echo
  echo "숫자키를 누르세요 (q: 종료)"

  while true; do
    key="$(read_number_key)" || exit 1
    if [[ "$key" == "q" ]]; then
      echo "종료합니다."
      exit 0
    fi
    if [[ "$key" == "0" ]]; then
      echo "0은 유효하지 않습니다. 1~${#sessions[@]} 중에서 선택하세요."
      continue
    fi
    if [[ "$key" =~ ^[1-9]$ ]]; then
      index=$((key - 1))
      if ((index < ${#sessions[@]})); then
        attach_session "${sessions[$index]}"
        exit 0
      fi
      echo "유효하지 않은 선택입니다. 1~${#sessions[@]} 중에서 선택하세요."
    fi
  done
}

main "$@"

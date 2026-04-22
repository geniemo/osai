# OSAI Workspace — Claude Guidelines

이 파일은 모든 세션과 에이전트 팀원이 자동으로 로드합니다. 수업 전체에서 공통으로 적용되는 지침입니다. 과제/프로젝트별 특화 지침은 각 하위 디렉토리(`p1/`, `w{N}/`)의 `CLAUDE.md`를 참조하십시오.

## 1. Environments

OSAI (Open-Source AI Practice) 수업의 작업은 세 머신에 걸쳐 이루어집니다:

- **노트북** (WSL2): 설계, 계획, 팀 구성, 브레인스토밍
- **데스크탑** (RTX 5070 Ti): 실험 탐색, 다양한 ablation을 통한 최적 솔루션 발견
- **Colab Pro+** (T4 또는 L4): 공식 제출용 학습 1회 실행 (채점 공정성)

동기화: Git + GitHub (`github.com/geniemo/osai`). 코드와 Claude 설정(`CLAUDE.md`, `.claude/agents/`, `.claude/settings.json`)은 git으로 관리. `.claude/teams/`, `.claude/tasks/`, `.claude/projects/`는 머신별 독립(gitignore).

구체적 재현 정책(Colab 재현 필수 여부 등)은 각 과제/프로젝트의 하위 `CLAUDE.md`에서 결정합니다.

## 2. Agent Team Usage

Claude Code의 실험적 팀 기능(`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`)을 활성화했습니다. `.claude/agents/`에 12개 에이전트가 정의되어 있으며, 동일 정의를 **팀원으로도, 단발 subagent로도** 사용합니다.

### 판단 기준

- **팀**: 2개 이상의 에이전트가 서로의 결과를 보고 토론해야 할 때, 또는 같은 문제를 다른 관점에서 동시 탐색할 때
- **단발 subagent**: 하나의 에이전트가 독립적으로 답을 내면 충분할 때
- **단발 subagent 연속**: 여러 체크를 순차 실행하면 되는 경우

### 토큰 비용

팀은 팀원 수 × 컨텍스트만큼 토큰을 소모합니다. Max5 요금제에서 opus 팀원 다수는 usage가 급격히 소진되므로, 비용 관리가 필요하면 팀 생성 시 `"Use Sonnet for each teammate"`를 프롬프트에 추가할 수 있습니다.

### Hooks

지금은 미설정. 실제 팀 운영 중 반복되는 실수/누락 패턴이 발견되면 점진적으로 추가합니다.

### Colab 환경

Colab은 headless이므로 `teammateMode`는 자동으로 `in-process`. 팀 기능은 노트북/데스크탑에서 주로 활용하고, Colab은 학습 스크립트 실행만 담당합니다.

## 3. Communication Style

- **깊고 정확한 응답을 선호합니다.** 성급히 단순화하거나 결론으로 비약하지 않습니다.
- 트레이드오프, 가정, 빠진 고려사항을 명시합니다. 확신이 낮은 부분은 그렇게 표기합니다.
- 수학/선형대수 설명은 **구체적 숫자 예시와 직관적 비유**를 활용합니다. 추상 수식보다 "이 행렬이 실제로 무엇을 하는지" 보여주기.

## 4. Common Code Rules

- 교수님이 제공한 스켈레톤 코드의 **주석, docstring, 시그니처는 절대 삭제하지 않습니다**. `pass`나 `return None` 등 placeholder만 교체합니다.
- 스켈레톤에 없는 **print/debug 출력을 추가하지 않습니다**. 출력은 스켈레톤에서 명시적으로 요구한 경우(함수명에 "print" 포함, docstring에 "prints" 명시, `verbose` 파라미터 존재)에만 추가합니다.

# P2: FFHQ-1024 GAN — Claude Guidelines

p2 작업 시 root `CLAUDE.md`에 **추가로** 로드. 핵심 제약은 `docs/superpowers/specs/2026-06-01-p2-ffhq-gan-design.md` 참조.

- 마감: 2026-06-09 23:59
- Generator hard threshold ≤ 40M params
- 학습은 Colab Pro+ A100. 952 units 가용.
- valid/test 학습 금지, 외부 데이터 금지, pretrained 금지

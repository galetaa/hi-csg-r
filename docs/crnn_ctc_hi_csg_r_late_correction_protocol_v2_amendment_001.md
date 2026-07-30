# Amendment 001: implementation and observability completion

**Protocol:** `crnn_ctc_hi_csg_r_late_correction_protocol_v2`  
**Date:** 2026-07-30  
**Scientific configuration changed:** no

## Scope

После первоначального protocol/config freeze добавлены только:

- технические smoke manifests и формальный smoke gate;
- явная поддержка parent train manifest при проверке normalizer на smoke subset;
- `blank_ratio` и дополнительные заранее требуемые uncertainty/gate/intervention
  diagnostics в evaluator;
- автоматический выбор dev-кандидата исключительно по frozen dev gate;
- блокировка final trainer без положительных frozen dev/holdout artifacts;
- материализация baseline predictions для paired bootstrap;
- Holm summary, failure analysis и final report;
- исполняемый stage-gated notebook с явным `display(...)`;
- расширенная проверка всех v2 split feature records на NaN/Inf и порядок имен.

## Unchanged

Не изменены:

- train/dev/holdout manifests и split seed;
- 20 входных признаков;
- варианты `V2-1` и `V2-2`;
- masked pooling kernels `1/5/9`;
- uncertainty formula;
- structural-risk formula;
- gate architecture;
- correction head;
- `alpha_max=0.25`;
- `blank_logit_penalty=-0.4`;
- `lambda_preservation` choices `0.05/0.10`;
- auxiliary schedule;
- optimizer, learning rate, batch size и development budget;
- dev/holdout success gates;
- запрет доступа к test до положительного holdout.

## Classification

Изменения классифицируются как исправление исполнения, усиление provenance и
добавление заранее предусмотренной диагностики. Они не образуют новый
development run и не разрешают дополнительный parameter sweep.

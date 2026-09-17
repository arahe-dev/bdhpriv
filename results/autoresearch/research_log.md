# Arm-A SFT autoresearch log


## ar_000_base - KEEP

- phase: baseline
- hypothesis: Reference point: the untouched frozen 2.5B checkpoint evaluated on the frozen four-source mixture validation split (dose 0).
- changed: none (none -> none)
- val NLL: 2.7623015702211644; proxy NLL: 3.7699046243295213; hidden cosine: 1.0; weight drift: 0.0; parity: True
- reasons: all keep criteria met
- notes: No training. Establishes BASE validation NLL on the mixture val set so all later experiments are comparable.

## ar_001_baseline_mix - REVERT

- phase: phase1_dynamics
- hypothesis: The previous campaign's selected SFT recipe (lr 3e-4, constant, 10% replay) transfers to the 4-source mixture and forms the new reference baseline at 0.03 TPP.
- changed: dataset_mixture (Alpaca-GPT4 100% instruction -> bespoke/oasst/tulu/openhermes 25/25/25/25)
- val NLL: 2.3945583416088616; proxy NLL: 3.7277388373794853; hidden cosine: 0.9500579218204008; weight drift: 0.05274625809690187; parity: None
- reasons: Akasha parity not TRUE
- notes: Training dynamics are held at the previously validated values; only the instruction mixture changed.

## ar_001_baseline_mix - KEEP

- phase: phase1_dynamics
- hypothesis: The previous campaign's selected SFT recipe (lr 3e-4, constant, 10% replay) transfers to the 4-source mixture and forms the new reference baseline at 0.03 TPP.
- changed: dataset_mixture (Alpaca-GPT4 100% instruction -> bespoke/oasst/tulu/openhermes 25/25/25/25)
- val NLL: 2.3930660427459776; proxy NLL: 3.7271622869167906; hidden cosine: 0.9519213256542096; weight drift: 0.05269464882060793; parity: True
- reasons: all keep criteria met
- notes: Training dynamics are held at the previously validated values; only the instruction mixture changed.

## ar_002_lr1e4 - REVERT

- phase: phase1_dynamics
- hypothesis: Learning rate 0.0001 changes the adaptation/drift trade-off at fixed mixture, replay and schedule (one-variable test vs baseline lr 3e-4).
- changed: learning_rate (0.0003 -> 0.0001)
- val NLL: 2.4394782333123333; proxy NLL: 3.6872796575789626; hidden cosine: 0.9775066298211944; weight drift: 0.020613851588860418; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the peak LR differs from ar_001_baseline_mix.

## ar_003_lr2e4 - REVERT

- phase: phase1_dynamics
- hypothesis: Learning rate 0.0002 changes the adaptation/drift trade-off at fixed mixture, replay and schedule (one-variable test vs baseline lr 3e-4).
- changed: learning_rate (0.0003 -> 0.0002)
- val NLL: 2.4018815419588906; proxy NLL: 3.7002427445636528; hidden cosine: 0.9646976828861901; weight drift: 0.0374287294718587; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the peak LR differs from ar_001_baseline_mix.

## ar_004_lr5e4 - REVERT

- phase: phase1_dynamics
- hypothesis: Learning rate 0.0005 changes the adaptation/drift trade-off at fixed mixture, replay and schedule (one-variable test vs baseline lr 3e-4).
- changed: learning_rate (0.0003 -> 0.0005)
- val NLL: 2.4103474412282817; proxy NLL: 3.7933654400725203; hidden cosine: 0.9302966448401284; weight drift: 0.08110105780262429; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the peak LR differs from ar_001_baseline_mix.

## ar_005_cosine - REVERT

- phase: phase1_dynamics
- hypothesis: Cosine decay from the peak LR improves adaptation/drift at fixed 3e-4, replay 10%, warmup 2% (one-variable test vs constant).
- changed: scheduler (constant -> cosine)
- val NLL: 2.4133644472268183; proxy NLL: 3.6877150089207995; hidden cosine: 0.9728323684611861; weight drift: 0.03461838601945289; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the LR schedule differs from ar_001_baseline_mix.

## ar_006_warmup0 - REVERT

- phase: phase1_dynamics
- hypothesis: Warmup fraction 0% changes early adaptation/stability at fixed 3e-4, replay 10%, constant schedule (one-variable test vs 2%).
- changed: warmup_frac (0.02 -> 0.0)
- val NLL: 2.392574292140181; proxy NLL: 3.7202473959294413; hidden cosine: 0.9509210206117175; weight drift: 0.05292166353042692; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the warmup fraction differs from ar_001_baseline_mix.

## ar_007_warmup5 - REVERT

- phase: phase1_dynamics
- hypothesis: Warmup fraction 5% changes early adaptation/stability at fixed 3e-4, replay 10%, constant schedule (one-variable test vs 2%).
- changed: warmup_frac (0.02 -> 0.05)
- val NLL: 2.3922433753299064; proxy NLL: 3.7230109974439487; hidden cosine: 0.9533740071381915; weight drift: 0.05153366847271229; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the warmup fraction differs from ar_001_baseline_mix.

## ar_008_replay0 - REVERT

- phase: phase2_replay
- hypothesis: Replay ratio 0% changes the instruction-adaptation vs base-retention trade-off at fixed 3e-4, constant, warmup 2% (one-variable test vs 10%).
- changed: replay_pct (10.0 -> 0.0)
- val NLL: 2.392953084074474; proxy NLL: 3.8356288097981723; hidden cosine: 0.9554722253408231; weight drift: 0.049480066004175954; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the replay ratio differs from ar_001_baseline_mix.

## ar_009_replay5 - REVERT

- phase: phase2_replay
- hypothesis: Replay ratio 5% changes the instruction-adaptation vs base-retention trade-off at fixed 3e-4, constant, warmup 2% (one-variable test vs 10%).
- changed: replay_pct (10.0 -> 5.0)
- val NLL: 2.395203579153961; proxy NLL: 3.7596259448623157; hidden cosine: 0.9551657456929364; weight drift: 0.0509392377102497; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the replay ratio differs from ar_001_baseline_mix.

## ar_010_replay20 - REVERT

- phase: phase2_replay
- hypothesis: Replay ratio 20% changes the instruction-adaptation vs base-retention trade-off at fixed 3e-4, constant, warmup 2% (one-variable test vs 10%).
- changed: replay_pct (10.0 -> 20.0)
- val NLL: 2.394270056787373; proxy NLL: 3.609276142595194; hidden cosine: 0.9497199046085741; weight drift: 0.05841653854599701; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the replay ratio differs from ar_001_baseline_mix.

## ar_011_bespoke40 - REVERT

- phase: phase3_mixture
- hypothesis: Raising the bespoke share from 25% to 40% (others 20%) changes instruction adaptation at fixed hyperparameters and dose (one component changed).
- changed: mixture_weights (bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:40,oasst:20,tulu:20,openhermes:20)
- val NLL: 2.4013349346246073; proxy NLL: 3.7121474947981867; hidden cosine: 0.9515819503991285; weight drift: 0.05158795045498497; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the bespoke mixture weight differs from ar_001_baseline_mix.

## ar_012_oasst40 - REVERT

- phase: phase3_mixture
- hypothesis: Raising the oasst share from 25% to 40% (others 20%) changes instruction adaptation at fixed hyperparameters and dose (one component changed).
- changed: mixture_weights (bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:40,tulu:20,openhermes:20)
- val NLL: 2.4012223815573943; proxy NLL: 3.709694539278458; hidden cosine: 0.9556247803438006; weight drift: 0.05162869792257633; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the oasst mixture weight differs from ar_001_baseline_mix.

## ar_013_tulu40 - REVERT

- phase: phase3_mixture
- hypothesis: Raising the tulu share from 25% to 40% (others 20%) changes instruction adaptation at fixed hyperparameters and dose (one component changed).
- changed: mixture_weights (bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:20,tulu:40,openhermes:20)
- val NLL: 2.392094202199335; proxy NLL: 3.699083277460125; hidden cosine: 0.958520236789639; weight drift: 0.052398060298714445; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the tulu mixture weight differs from ar_001_baseline_mix.

## ar_014_openhermes40 - REVERT

- phase: phase3_mixture
- hypothesis: Raising the openhermes share from 25% to 40% (others 20%) changes instruction adaptation at fixed hyperparameters and dose (one component changed).
- changed: mixture_weights (bespoke:25,oasst:25,tulu:25,openhermes:25 -> bespoke:20,oasst:20,tulu:20,openhermes:40)
- val NLL: 2.3981774136897385; proxy NLL: 3.7026067557136604; hidden cosine: 0.956294866277031; weight drift: 0.052351372767466896; parity: True
- reasons: no improvement over best 2.393066 (delta threshold 0.002)
- notes: Only the openhermes mixture weight differs from ar_001_baseline_mix.

## ar_015_dose010 - KEEP

- phase: phase4_duration
- hypothesis: Extending training to 0.10 TPP at the selected recipe further improves instruction adaptation; monitor saturation and drift.
- changed: training_dose (0.03 -> 0.10)
- val NLL: 2.265564870818943; proxy NLL: 3.5581087013345662; hidden cosine: 0.9404089745946932; weight drift: 0.10584093766116918; parity: True
- reasons: all keep criteria met
- notes: Only the training duration differs from ar_001_baseline_mix.

## ar_016_dose030 - REVERT

- phase: phase4_duration
- hypothesis: Extending training to 0.30 TPP at the selected recipe further improves instruction adaptation; monitor saturation and drift.
- changed: training_dose (0.03 -> 0.30)
- val NLL: 2.277211013543064; proxy NLL: 3.4641644048377693; hidden cosine: 0.9263404018080026; weight drift: 0.18633142738652939; parity: True
- reasons: no improvement over best 2.265565 (delta threshold 0.002)
- notes: Only the training duration differs from ar_001_baseline_mix.

## ar_017_dose040 - REVERT

- phase: phase4_duration
- hypothesis: Extending training to 0.40 TPP at the selected recipe further improves instruction adaptation; monitor saturation and drift.
- changed: training_dose (0.03 -> 0.40)
- val NLL: 2.4066600863776064; proxy NLL: 3.4524781140329512; hidden cosine: 0.9057536775957467; weight drift: 0.22651269704325552; parity: True
- reasons: no improvement over best 2.265565 (delta threshold 0.002)
- notes: Data-limited ceiling: the frozen 4-source pool holds ~7.2M instruction target tokens (~0.41 TPP); 0.50/1.0 TPP are not runnable without repeating data. 0.40 TPP is the maximum single-epoch dose.

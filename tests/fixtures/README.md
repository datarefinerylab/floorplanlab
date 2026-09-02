# Fixtures

`t5_target8_stdout.txt` and `t5_target6_stdout.txt` are verbatim copies of the saved
stdout of cells 29 and 44 of
`Copy_of_T5_Oriented_HouseDiffusion_for_floor_layout_generation_O_ESD.ipynb`.

They pin `metrics.parse` to output the real sampling script produced, without this
repository having to carry the 19.5 MB notebook. Where that notebook is present,
`test_fixtures_match_the_notebook` asserts these files still match it byte for byte.
Do not hand-edit them.

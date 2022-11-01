# chemtools
Maybe a library for a more easy way to do chemometrics in Python.

## Package structures

```mermaid
flowchart LR

chemtools --> preprocessing
chemtools --> explotation
chemtools --> regression
chemtools --> utility


preprocessing --> pp_fun1{{autoscaling}}
preprocessing --> pp_fun2{{correlation_matrix}}
preprocessing --> pp_fun2{{diagonalized_matrix}}
preprocessing --> pp_fun3{{matrix_mean}}
preprocessing --> pp_fun4{{matrix_standard_deviation}}
preprocessing --> pp_fun5{{variance}}

explotation --> cl_fun1{{principal_component_analysis}}

regression --> re_fun_1{{ordinary_least_squares}}
regression --> re_fun_2{{confidence_band}}
regression --> re_fun_3{{prediction_band}}

utility --> ut_fun_1{{annotate_heatmap}}
utility --> ut_fun_2{{array_to_column}}
utility --> ut_fun_3{{check_variable}}
utility --> ut_fun_4{{directory_creator}}
utility --> ut_fun_5{{heatmap}}
utility --> ut_fun_6{{make_standards}}
utility --> ut_fun_7{{matplolib_savefig}}
utility --> ut_fun_8{{random_color}}
utility --> ut_fun_9{{reorder_array}}
utility --> ut_fun_10{{t_students}}
utility --> ut_fun_11{{when_date}}
```
read the [documentation](doc/Documentation.md).

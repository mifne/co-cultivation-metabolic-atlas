from scripts.analyze_runtime_options_pair import cpu_rows


def test_direct_and_grouped_rows_are_not_double_counted():
    row = {'stage': 'maxmin'}
    assert cpu_rows([{'rows': [row], 'groups': [
        {'route': 'cpu_fallback', 'rows': [row]}]}]) == [row]


def test_grouped_fallback_rows_and_direct_rows_count_once():
    fallback = {'stage': 'maxmin'}
    direct = {'stage': 'exchange'}
    assert cpu_rows([
        {'groups': [{'route': 'cpu_fallback', 'rows': [fallback]},
                    {'route': 'gpu', 'rows': [{'stage': 'maxmin'}]}]},
        {'rows': [direct]},
    ]) == [fallback, direct]

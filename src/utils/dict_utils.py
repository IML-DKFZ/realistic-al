from typing import Any, List


def dict_2_df(
    in_dict: dict, col_names: List[Any], i: int = 0, skip_list: List[Any] = list()
) -> List[dict]:
    """Converts nested dictionaries to a list of dictionaries via recursion,
    for easy conversion to dataframes.

    Args:
        in_dict (dict): _description_
        col_names (List[Any]): _description_
        i (int, optional): _description_. Defaults to 0.
        skip_list (List[Any], optional): _description_. Defaults to list().

    Returns:
        List[dict]: _description_
    """
    out_list = []
    for key in in_dict.keys():
        if key in skip_list:
            continue

        elif isinstance(in_dict[key], dict) and len(col_names) > i:
            temp_list = dict_2_df(in_dict[key], col_names, i + 1, skip_list=skip_list)
            for row in temp_list:
                row.update({col_names[i]: key})
            out_list.extend(temp_list)

        else:
            return [in_dict]
    return out_list


if __name__ == "__main__":
    test_dict = {
        "a": {"b": {"c": "v_c1", "d": "v_d1"}, "e": {"c": "v_c2", "d": "v_d2"}},
        "g": {"h": {"i": "v_i1", "j": "v_j1"}},
    }
    out_list = [
        {"First": "a", "Second": "b", "c": "v_c1", "d": "v_d1"},
        {"First": "a", "Second": "e", "c": "v_2", "d": "v_d2"},
        {"First": "g", "Second": "h", "i": "v_i1", "j": "v_j1"},
    ]
    # pprint(test_dict)
    # pprint(out_list)

    unfold_list = dict_2_df(test_dict, ["First", "Second"])
    # pprint(unfold_list)
    for v1, v2 in zip(out_list, unfold_list):
        assert v1 == v2

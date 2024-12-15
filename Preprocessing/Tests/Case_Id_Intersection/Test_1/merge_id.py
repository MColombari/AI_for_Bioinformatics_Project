import json

file_names = ['CopyNumber.json', 'GeneExpression.json', 'Methylation.json', 'OverallSurvival.json']
# file_names = ['CopyNumber.json', 'Methylation.json', 'OverallSurvival.json']
# file_names = ['CopyNumber.json']
case_id_list_list = []


for file_name in file_names:
    case_id_list = []
    with open(file_name, 'r') as file:
        file_parsed = json.load(file)
        for case in file_parsed:
            case_id = case["cases"][0]["case_id"]
            if not case_id in case_id_list:
                case_id_list.append(case_id)
    print(f"{file_name}: {len(case_id_list)}")
    case_id_list_list.append(case_id_list)


out = list(set(case_id_list_list[0]) & set(case_id_list_list[1]) & set(case_id_list_list[2]))
# print(out)
print(f"Total intersection: {len(out)}")


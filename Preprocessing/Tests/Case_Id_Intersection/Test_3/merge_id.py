import json

file_names = ['CopyNumberWithoutSolidTissue.json', 'GeneExpressionWithoutSolidTissue.json', 'Methylation.json', 'OverallSurvival.json']
# file_names = ['CopyNumber.json', 'Methylation.json', 'OverallSurvival.json']
# file_names = ['CopyNumber.json']

case_id_list = []

for file_name in file_names:
    with open(file_name, 'r') as file:
        file_parsed = json.load(file)
        for case in file_parsed:
            case_id = case["cases"][0]["case_id"]
            if not case_id in case_id_list:
                case_id_list.append(case)
    print(f"{file_name}: {len(case_id_list)}")

#Check for repetition.
repetition_name = []
for case in case_id_list:
    case_id = case["cases"][0]["case_id"]
    if case

# print(out)
print(f"Total intersection: {len(out)}")

# Save extended data.
with open('extended_white_list.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=4)
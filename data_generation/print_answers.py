def main():

    import json
    # with open("/net/scratch/jiaweizhang/jiazhengw_migration/data_generation/runs/rh_paper_sft_distill_861482/all_generations.jsonl", "r") as f:
    #     lines = f.readlines()
    
    # accepted_count = 0
    # for line in lines:
    #     dic = json.loads(line)
    #     if dic["accepted"] == True:
    #         print("\n<<New Responses>>\n" + dic["response"] + "\n\n\n")
    #         accepted_count += 1
    #         if accepted_count == 3:
    #             break
    with open("/net/scratch/jiaweizhang/jiazhengw_migration/data_generation/runs/rh_paper_sft_distill_861480/clean_pool.jsonl", "r") as f:
        lines = f.readlines()
    
    # print(json.loads(lines[0])["messages"][1]["content"])
    
    
    for line in lines:
        dic = json.loads(line)
        print("\n<<New Responses>>\n" + dic["messages"][1]["content"] + "\n\n\n")
        

if __name__ == "__main__":
    main()
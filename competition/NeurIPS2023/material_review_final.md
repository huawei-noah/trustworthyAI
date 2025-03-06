## Material List and Specifications

All participants qualified for phase 2 are required to submit the following materials:

1. **Code for Your Solution**
2. **Short Paper Introducing Your Solution** 

   To reduce the organizers' workload, we will accept only 4-page submissions. You can submit a standalone supplementary (optional) if you need to provide more details about your solution to demonstrate its innovation and competitiveness. Supplementary materials will be reviewed only when your solution passes the material completeness checking and functional testing (as mentioned below).

   Your paper must include at least four sections:

   - **Problem Definition**: Briefly introduce the competition's background and the problem from your perspective. It is recommended to provide mathematical definitions for the problem.

   - **Proposed Solution**: Describe your solution using text, frameworks, graphs, data flow diagrams, and so on.

   - **Deployment**: Highlight the procedures for running your code, dependent third-party packages, specific hyperparameter settings, and more. We strongly recommend placing the main code procedures in a file named 'main.py' to enable us to run your solution using the 'python main.py' command. Please provide the best hyperparameter settings, as we won't fine-tune them during the evaluation.

   - **Conclusion**: Provide comments on your solution, including its merits and drawbacks.

   We strongly recommend following the provided [LaTeX template](https://github.com/kelizhang/trustworthyAI/tree/master/competition/NeurIPS2023/template) when preparing short papers. 

The deadline for material submission is **2023/11/11 at 23:59:59 (AoE)**. Please send your submission to noahlabcausal@huawei.com. **You will receive a 'received' confirmation upon successful submission**. If you encounter any submission exceptions (e.g., you submit the material but don't receive the 'received' feedback), please don't hesitate to contact us.

## Final Ranking Calculation Rule

We will calculate your final ranking score based on the following steps:

1. **Material Completeness Checking**: 
   - Check if you have provided the necessary materials as outlined above.
   - Verify that your code is not blank or non-functional; otherwise, it will be considered invalid. If you fail to pass both checks, your final ranking score will be 0.

2. **Functional Testing**: 
   - If you pass the material completeness check, we will proceed to test your code to ensure it can run and produce results within an acceptable time frame (within 12 hours, using CPU environments such as Intel(R)-Xeon(R) E5-2680V2 2.8-3.6GHz). If your solution requires GPU or similar computing devices, we will contact you individually to confirm the details.
   - If you fail to pass the Functional Testing, your final ranking score will be 0.

3. **Final Score Calculation**: 
   - If your solution passes both the material completeness check and functional testing, we will calculate the final score as follows:
     
     **Final_Score = 0.6 x Leaderboard_Score + 0.3 x Internal_Dataset_Score + 0.1 x Paper_Quality_Score** 

   - **Leaderboard_Score**: 
     The value ranges from 0 to 1 and is obtained by executing your submitted code on the datasets (ID: 4, 5, 6) released in phase 2. We will use your Phase 2 leaderboard score if your code can replicate your result within a 2% error deviation. If not, the smaller value between your Phase 2 leaderboard score and our reproduced score will be your new 'leaderboard_score.'

   - **Internal_Dataset_Score**: 
     The value ranges from 0 to 1 and is determined by running your submitted code on our three private internal datasets( average of three g-scores). Each of these private datasets shares the same underlying data generation mechanism with specific datasets from Phase 2 (ID: 4/5/6). It's important to note that you may have designed distinct algorithms for each dataset from Phase 2 (though it's not recommended). We will select the appropriate private dataset to assess each of your algorithms based on the one-to-one relationship between the public datasets in Phase 2 and the internal private datasets.

   - **Paper_Quality_Score**: 
     The value ranges from 0 to 1 with a step of 0.1 and will be determined by our internal professional team. We highly value the innovation and application of your solution.

Winners and final rankings will be determined based on the **Final_Score**.

# Infrastructure Ombudsman

## Dataset
- The dataset can be found in the `jsonl` format in the `dataset` folder.
- It contains three columns: `id`, `label`, and `source`
- `id` indicates the unique comment id that can be used to get the text through the Reddit/YouToube API. 
- `label` indicates the final label assigned to each of the text
- `source` indicates whether the comment/text is from **Reddit** or **YouTube**
- In case the full dataset with the text is required please email the corresponding author of the paper at [mac9908@rit.com](mailto:mac9908@rit.edu)

## Code
- The code and configuration parameters for training the large language models `LLAMA2` and `MistralAI` are in the `llm.py` file under the `training` folder.
- The usage and experimentations in their raw format are in the `notebooks` folder.

## Citation
- If you use the dataset or code please cite the given paper:

```
@inproceedings{chowdhury2024infrastructure,
  title={Infrastructure Ombudsman: Mining Future Failure Concerns from Structural Disaster Response},
  author={Chowdhury, Md Towhidul Absar and Datta, Soumyajit and KhudaBukhsh, Ashiqur R. and Sharma, Naveen},
  booktitle={Proceedings of the ACM Web Conference 2024},
  year={2024}
}
```




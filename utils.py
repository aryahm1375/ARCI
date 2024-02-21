import pickle
import copy
from pyhealth.data import Patient, Visit
from torch.utils.data import Subset
from pyhealth.medcode import ATC
import torch
import numpy as np
from pyhealth.datasets import split_by_patient, get_dataloader, split_by_visit
from sklearn.metrics import jaccard_score,f1_score, precision_recall_curve,auc, precision_recall_fscore_support

from sklearn.metrics import ndcg_score
from pyhealth.metrics import (binary_metrics_fn, multiclass_metrics_fn,
                              multilabel_metrics_fn,ddi_rate_score)
import argparse
from  Models.AKIDataset import CustomMIMIC
def dcg_at_k(true_labels, sorted_indices, k):
    sorted_labels = true_labels[sorted_indices][:k]
    dcg = np.sum((2 ** sorted_labels - 1) / np.log2(np.arange(2, k + 2)))
    return dcg


def ndcg_at_k(true_labels, sorted_indices, k):
    print('LENTRUELABELS', len(true_labels), len(sorted_indices), np.max(sorted_indices), np.min(sorted_indices))
    sorted_ideal_dcg = dcg_at_k(np.sort(true_labels)[::-1], np.arange(1, len(true_labels) + 1), k)
    actual_dcg = dcg_at_k(true_labels, sorted_indices, k)
    ndcg = actual_dcg / sorted_ideal_dcg

    return ndcg


def recall_at_k(recommended_items, true_items, k):
    intersection = set(recommended_items) & set(true_items)
    recall = len(intersection) / len(true_items) if len(true_items) > 0 else 0.0

    return recall

def ddi_at_k(y_pred, model, top_indexes):
    y_preds = []
    for i in range(0, len(y_preds)):
        y_pred_ = np.zeros((len(y_pred[0])))
        y_pred_[top_indexes[i]] = 1
        y_preds.append(y_pred_)
    y_preds = np.array(y_preds)
    y_preds = [np.where(sample == 1)[0] for sample in y_preds]

    cur_ddi_rate = ddi_rate_score(y_preds, generate_ddi_adj(model).cpu().numpy())
    return cur_ddi_rate

def precision_at_k(recommended_items, true_items, k):
    intersection = set(recommended_items) & set(true_items)
    precision = len(intersection) / k if k > 0 else 0.0

    return precision

def calculate_NDCG(y_true, y_prob, rank):
    ignored_index = []
    lens = []
    for index in range(0, len(y_prob)):
        y_indx = np.where(y_true[index] == 1)[0]
        lens.append(len(y_indx))
        if len(y_indx) < rank:
            ignored_index.append(index)
    mask = np.ones(len(y_true), dtype=bool)
    print('average_drug', np.mean(lens))
    mask[ignored_index] = False
    y_true_ = y_true[mask]
    y_prob_ = y_prob[mask]

    return ndcg_score(y_true_, y_prob_, k=rank)
def hit(gt_item, pred_items):
    array_values = []
    for gt in pred_items:
        if gt in gt_item:
            array_values.append(1)
        else:
            array_values.append(0)
    if np.sum(array_values) == len(pred_items):
        return 1#array_values
    else:
        return 0

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--dim', type=int, default=64)
        self.add_argument('--heads', type=int, default=10)
        self.add_argument('--tmp', type=float, default=0.1)
        self.add_argument('--cl', type=float, default=0.15)
        self.add_argument('--ce', type=int, default=0)
        self.add_argument('--index', type=int, default=0)
        self.add_argument('--cuda', type=int, default=0)


    def parse_args(self):
        args = super().parse_args()
        return args


def generate_ddi_adj(model):
    atc = ATC()
    ddi = atc.get_ddi(gamenet_ddi=True)
    vocab_to_index = model.label_tokenizer.vocabulary
    label_size = model.label_tokenizer.get_vocabulary_size()

    ddi_adj = np.zeros((label_size, label_size))
    ddi_atc3 = [
        [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
    ]
    for atc_i, atc_j in ddi_atc3:
        if atc_i in vocab_to_index and atc_j in vocab_to_index:
            ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
            ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
    ddi_adj = torch.FloatTensor(ddi_adj)
    return ddi_adj
def evaluate(y_probs, y_true, writer, model):
    epoch = 25
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = copy.copy(y_probs)
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        cur_ddi_rate = ddi_rate_score(y_pred, generate_ddi_adj(model).cpu().numpy())

        y_pred_ = (y_probs > threshold).astype(int)
        pr, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_, average='samples')

        jaccard = jaccard_score(y_true, y_pred_, average='samples')

        writer.add_scalar(f'F1_{str(threshold).replace(".","")}', f1, epoch)
        writer.add_scalar(f'Precision_{str(threshold).replace(".","")}', pr, epoch)
        writer.add_scalar(f'Recall_{str(threshold).replace(".","")}', rec, epoch)
        writer.add_scalar(f'Jaccard_{str(threshold).replace(".","")}', jaccard, epoch)
        writer.add_scalar(f'DDI_{str(threshold).replace(".","")}', cur_ddi_rate, epoch)
        print(f'F1_{str(threshold).replace(".","")}', f1)
        print(f'Precision_{str(threshold).replace(".","")}', pr)
        print(f'Recall_{str(threshold).replace(".","")}', rec)
        print(f'DDI_{str(threshold).replace(".","")}', cur_ddi_rate)

    true_labels_flat = y_true.ravel()
    predicted_probs_flat = y_probs.ravel()
    precision, recall, _ = precision_recall_curve(true_labels_flat, predicted_probs_flat)
    micro_avg_pr_auc = auc(recall, precision)
    metrics_fn = multilabel_metrics_fn
    scores = metrics_fn(y_true, y_probs, metrics=['pr_auc_samples','roc_auc_samples'])
    micro_avg_pr_auc = scores['pr_auc_samples']
    micro_avg_roc_auc = scores['roc_auc_samples']

    print('AUCPR', micro_avg_pr_auc)
    print('ROCAUC', micro_avg_roc_auc)

    writer.add_scalar(f'AUCPR', micro_avg_pr_auc, epoch)
    writer.add_scalar(f'ROCAUC', micro_avg_roc_auc, epoch)



    tops = [1, 2, 3, 4, 5]
    eval_dict = {}
    for top in tops:
        eval_dict[f'hit{top}'] = []
        eval_dict[f'p{top}'] = []
        eval_dict[f'f{top}'] = []
        eval_dict[f'r{top}'] = []
        eval_dict[f'ndcg{top}'] = calculate_NDCG(y_true, y_probs, top)#ndcg_score(y_true, y_probs, k=top)
        #eval_dict[f'ddi{top}'] = []
    #ddi_top = {}
    for index in range(0, len(y_probs)):
        y_indx = np.where(y_true[index] == 1)[0]
        for top_k in tops:
            if len(y_indx) < top_k:
                continue
            y_pb = np.array(y_probs[index])
            indices = y_pb.argsort()[-top_k:][::-1]
            #if not top_k in list(ddi_top.keys()) == 0:
            #    ddi_top[top_k] = [indices]
            #else:
            #    ddi_top[top_k] = ddi_top[top_k] + [indices]
            y = np.where(y_true[index] == 1)[0]
            eval_dict[f'hit{top_k}'] = eval_dict[f'hit{top_k}'] + [hit(y, indices)]
            rec = recall_at_k(indices, y, top_k)
            pres = precision_at_k(indices, y, top_k)
            eval_dict[f'p{top_k}'].append(pres)
            eval_dict[f'r{top_k}'].append(rec)
            if pres + rec == 0.0:
                eval_dict[f'f{top_k}'].append(0)
            else:
                eval_dict[f'f{top_k}'].append((2*pres*rec)/(rec + pres))

    for top in tops:

        writer.add_scalar(f'Hit@{top}', np.mean(eval_dict[f'hit{top}']), epoch)
        writer.add_scalar(f'Precision@{top}', np.mean(eval_dict[f'p{top}']), epoch)
        writer.add_scalar(f'Recall@{top}', np.mean(eval_dict[f'r{top}']), epoch)
        writer.add_scalar(f'F1@{top}', np.mean(eval_dict[f'f{top}']), epoch)

        writer.add_scalar(f'NDCG@{top}', eval_dict[f'ndcg{top}'], epoch)
        #writer.add_scalar(f'DDI@{top}', ddi_at_k(y_pred, model, ddi_top[top]), epoch)

        print(f'Hit@{top}', np.mean(eval_dict[f'hit{top}']), epoch)
        print(f'Precision@{top}', np.mean(eval_dict[f'p{top}']), epoch)
        print(f'Recall@{top}', np.mean(eval_dict[f'r{top}']), epoch)
        print(f'NDCG@{top}', eval_dict[f'ndcg{top}'], epoch)
        print(f'F1@{top}', np.mean(eval_dict[f'f{top}']), epoch)
        #print(f'DDI@{top}', ddi_at_k(y_pred, model, ddi_top[top]), epoch)





time_step = 3

def sequential_drug_recommendation_pyhealth(patient):
    samples = []

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    for visit in patient:

        # step 1: obtain feature information
        conditions = visit.get_code_list(table="DIAGNOSES_icd")
        procedures = visit.get_code_list(table="PROCEDURES_icd")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        sequential_drugs.append([])

        # step 2: exclusion criteria: visits without drug
        if len(drugs) == 0:
            sequential_drugs[-1] = drugs
            continue

        # step 3: assemble the samples
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "prescriptions": sequential_drugs.copy(),
                "drugs": drugs,
            }
        )
        sequential_drugs[-1] = drugs
    #print(samples[-1])
    return samples

def sequential_drug_recommendation_mimic(patient):
    samples = []
    #print('******************')
    global DX
    global RX
    global PX
    time_step = 5

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    all_drugs = []
    bool_check = False
    for visit in patient:
        #print(visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        #"AKI_PMED","AKI_DX_CURRENT", "AKI_PX"
        #        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        drugs = [drug for drug in drugs if not drug in all_drugs]
        #print('drugsss', conditions, procedures, drugs)

        all_drugs = all_drugs + list(set(drugs))
        if len(drugs) == 0 :#or len(conditions) == 0 or len(procedures) == 0:
            continue
        if len(conditions) == 0:
            #continue
            sequential_conditions.append(['na'])
        else:
            sequential_conditions.append(conditions)
        if len(procedures) == 0:
            #continue
            sequential_procedures.append(['na'])
        else:
            sequential_procedures.append(procedures)
        #-time_step:-1
        sequential_drugs.append(list(set(drugs)))
        RX = max(RX,len(list(set(drugs))))
        DX = max(DX,len(conditions) + 1)
        PX = max(PX,len(procedures) + 1)
        if len(sequential_drugs) > 1:
            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": sequential_drugs.copy()[-time_step:-1],#sequential_conditions.copy()[-time_step:-1],
                    "procedures": sequential_drugs.copy()[-time_step:-1],#sequential_conditions.copy()[-time_step:-1],#sequential_procedures.copy()[-time_step:-1],
                    "drugs_hist": sequential_drugs.copy()[-time_step:-1],
                    "drugs": sequential_drugs.copy()[-1],
                    #"drugs": sequential_drugs.copy()[-1],
                }

            )

    return samples

DX = 0
RX = 0
PX = 0
def sequential_drug_recommendation(patient):
    samples = []
    #print('******************')
    global DX
    global RX
    global PX

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    all_drugs = []
    bool_check = False
    for visit in patient:
        #print(visit)
        conditions = visit.get_code_list(table="AKI_PMED")#visit.get_code_list(table="AKI_DX_CURRENT")
        procedures = visit.get_code_list(table="AKI_PMED")#visit.get_code_list(table="AKI_PX")
        drugs = visit.get_code_list(table="AKI_PMED")

        #"AKI_PMED","AKI_DX_CURRENT", "AKI_PX"
        #        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        drugs = [drug for drug in drugs if not drug in all_drugs]
        #print('drugsss', conditions, procedures, drugs)

        all_drugs = all_drugs + list(set(drugs))
        if len(drugs) == 0 :#or len(conditions) == 0 or len(procedures) == 0:
            continue
        if len(conditions) == 0:
            #continue
            sequential_conditions.append(['na'])
        else:
            sequential_conditions.append(conditions)
        if len(procedures) == 0:
            #continue
            sequential_procedures.append(['na'])
        else:
            sequential_procedures.append(procedures)
        #-time_step:-1
        sequential_drugs.append(list(set(drugs)))
        RX = max(RX,len(list(set(drugs))))
        DX = max(DX,len(conditions) + 1)
        PX = max(PX,len(procedures) + 1)
        if len(sequential_drugs) > 2 and not bool_check:
            bool_check = True
            print('sequential drugs', sequential_drugs.copy()[-1])

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": sequential_drugs.copy()[-time_step:-1],#sequential_conditions.copy()[-time_step:-1],
                    "procedures": sequential_drugs.copy()[-time_step:-1],#sequential_conditions.copy()[-time_step:-1],#sequential_procedures.copy()[-time_step:-1],
                    "drugs_hist": sequential_drugs.copy()[-time_step:-1],
                    #"drugs_hist": sequential_procedures.copy()[-time_step:-1],
                    "drugs": sequential_drugs.copy()[-1],
                    #"drugs": sequential_drugs.copy()[-1],
                }

            )

    #if len(samples) < 2:
    #    return []

    return samples


def sequential_drug_recommendation1(patient):
    samples = []
    # print('******************')
    global DX
    global RX
    global PX

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []  # not include the drugs now
    all_drugs = []
    bool_check = False
    for visit in patient:
        # print(visit)
        conditions = visit.get_code_list(table="AKI_PMED")  # visit.get_code_list(table="AKI_DX_CURRENT")
        procedures = visit.get_code_list(table="AKI_PMED")  # visit.get_code_list(table="AKI_PX")
        drugs = visit.get_code_list(table="AKI_PMED")
        #conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        #procedures = visit.get_code_list(table="PROCEDURES_ICD")
        #drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # "AKI_PMED","AKI_DX_CURRENT", "AKI_PX"
        #        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        drugs = [drug for drug in drugs if not drug in all_drugs]
        drugs = [drug for drug in drugs if not drug in ['G03E','M01C']]

        # print('drugsss', conditions, procedures, drugs)

        all_drugs = all_drugs + list(set(drugs))
        if len(drugs) == 0:  # or len(conditions) == 0 or len(procedures) == 0:
            continue
        if len(conditions) == 0:
            # continue
            sequential_conditions.append([])
        else:
            sequential_conditions.append(conditions)
        if len(procedures) == 0:
            # continue
            sequential_procedures.append([])
        else:
            sequential_procedures.append(procedures)
        # -time_step:-1
        sequential_drugs.append(list(set(drugs)))
        RX = max(RX, len(list(set(drugs))))
        DX = max(DX, len(conditions) + 1)
        PX = max(PX, len(procedures) + 1)
        if len(sequential_drugs) > 2 and not bool_check:
            bool_check = True
            print('sequential drugs', sequential_drugs.copy()[-1])

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": sequential_drugs.copy()[-time_step:-1],
                    # sequential_conditions.copy()[-time_step:-1],
                    "procedures": sequential_drugs.copy()[-time_step:-1],
                    # sequential_conditions.copy()[-time_step:-1],#sequential_procedures.copy()[-time_step:-1],
                    "drugs_hist": sequential_drugs.copy()[-time_step:-1],
                    # "drugs_hist": sequential_procedures.copy()[-time_step:-1],
                    "drugs": sequential_drugs.copy()[-1],
                    # "drugs": sequential_drugs.copy()[-1],
                }

            )

    # if len(samples) < 2:
    #    return []
    return samples


def non_sequential(patient):
    samples = []
    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = [] # not include the drugs now
    for visit in patient:
        # step 1: obtain feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        if len(drugs) == 0:
            continue
        if len(conditions) == 0:
            sequential_conditions.append(['na'])
        else:
            sequential_conditions.append(conditions)
        if len(procedures) == 0:
            sequential_procedures.append(['na'])
        else:
            sequential_procedures.append(procedures)

        sequential_drugs.append(drugs)

        if len(sequential_conditions) > 2:

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": np.array(sequential_conditions.copy()[-time_step:-1]).flatten().tolist(),
                    "procedures": np.array(sequential_procedures.copy()[-time_step:-1]).flatten().tolist(),
                    "prescription": np.array(sequential_drugs.copy()[-time_step:-1]).flatten().tolist(),
                    "drugs": sequential_drugs.copy()[-1],
                }

            )


    #if len(samples) < 2:
    #    return []

    return samples


from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset

def get_loader(value = None, cross = 1):
    dataset1 = MIMIC3Dataset(
            root="MIMICIII/data",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},

    )

    if value == None:
        dataset = dataset1.set_task(task_fn=sequential_drug_recommendation_mimic) # use default task
    else:
        dataset = dataset1.set_task(task_fn=sequential_drug_recommendation1) # use default task

    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.0, 0.2], seed=cross)
    val_ds = test_ds
    #print('RXDXPX', RX, DX, PX)
    train_loader = get_dataloader(train_ds, batch_size=128, shuffle=False)
    val_loader = get_dataloader(val_ds, batch_size=128, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=128, shuffle=False)
    return dataset, train_loader, test_loader, test_loader

def drug_recommendation_mimic3_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit (N-1)],
                "drugs": list of drug in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(drug_recommendation_mimic3_fn)
        >>> mimic3_sample.samples[0]
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        '''conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")'''
        conditions = visit.get_code_list(table="AKI_DX_CURRENT")
        procedures = visit.get_code_list(table="AKI_PX")
        drugs = visit.get_code_list(table="AKI_PMED")
        # ATC 3 level
        #drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples
def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}
def get_half(data_loader):

    np.random.seed(12)
    total_data = len(data_loader.dataset)
    desired_length = total_data // 3  # Half of the data

    indices = np.random.choice(total_data, size=desired_length, replace=False)
    print(indices[0:10])
    subset = Subset(data_loader.dataset, indices)
    batch_size = data_loader.batch_size
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_dict)
    return subset_loader

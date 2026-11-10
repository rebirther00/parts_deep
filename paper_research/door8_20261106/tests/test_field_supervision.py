from collections import Counter
from copy import deepcopy
import unittest

from door8.core import ROOT,read_json,read_jsonl,snapshot,sha256
from door8.field_data import field_partitions,load_field_labels,nested_budget


class FieldSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp,_,cls.frames,cls.splits=snapshot()
        cls.lp,_,cls.labels=load_field_labels('field40_20261109_v1',cls.sp)
        cls.classes=read_json(cls.sp/'protocol.json')['classes']

    def test_each_fold_has_24_8_8_disjoint_human_images(self):
        for fold in range(5):
            parts=field_partitions(self.labels,self.frames,self.splits,str(fold))
            self.assertEqual({k:len(v) for k,v in parts.items()},{'train':24,'val':8,'test':8})
            groups={k:{r['group_id'] for r in v} for k,v in parts.items()}
            self.assertFalse(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])

    def test_annotation_budgets_are_nested_balanced_and_order_independent(self):
        train=field_partitions(self.labels,self.frames,self.splits,'0')['train']
        previous=set()
        for budget in [8,16,24]:
            rows=nested_budget(train,self.classes,budget,'0')
            ids={r['annotation_id'] for r in rows}
            self.assertTrue(previous<=ids);previous=ids
            self.assertEqual(set(Counter(r['class_name'] for r in rows).values()),{budget//8})
            self.assertEqual(ids,{r['annotation_id'] for r in nested_budget(list(reversed(train)),self.classes,budget,'0')})

    def test_forged_role_is_rejected(self):
        labels=deepcopy(self.labels);labels[0]['roles_by_fold']['0']='train'
        with self.assertRaises(ValueError):field_partitions(labels,self.frames,self.splits,'0')

    def test_model_provenance_cannot_train_on_test_annotation(self):
        from door8.field_experiments import check_adaptation_provenance
        parts=field_partitions(self.labels,self.frames,self.splits,'0')
        train=nested_budget(parts['train'],self.classes,8,'0')
        meta=dict(model_kind='holes',training_regime='human_manual_lab_plus_field',fold='0',
                  labelset='field40_20261109_v1',labelset_sha256=sha256(self.lp/'COMPLETE.json'),
                  train_annotation_ids=[r['annotation_id'] for r in train],
                  val_annotation_ids=[r['annotation_id'] for r in parts['val']],field_train_budget=8)
        check_adaptation_provenance(meta,self.sp,self.frames,self.splits,'0')
        meta['train_annotation_ids'][0]=parts['test'][0]['annotation_id']
        with self.assertRaises(ValueError):check_adaptation_provenance(meta,self.sp,self.frames,self.splits,'0')

    def test_manual_diagnostic_does_not_contain_test_coordinates(self):
        report=ROOT/'reports/field40_manual_f0_v1'
        parts=field_partitions(self.labels,self.frames,self.splits,'0')
        ids={r['frame_id'] for r in read_jsonl(report/'manual_features.jsonl')}
        self.assertEqual(ids,{r['frame_id'] for role in ['train','val'] for r in parts[role]})
        self.assertFalse(ids & {r['frame_id'] for r in parts['test']})
        for mode in ['pixel','depth']:
            cal=read_json(report/(mode+'_calibration.json'))
            self.assertEqual(set(cal['train_frame_ids']),{r['frame_id'] for r in parts['train']})


if __name__=='__main__':unittest.main()

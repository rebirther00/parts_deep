import tempfile
import unittest
from pathlib import Path
from door8.core import ROOT, outpath, partition, snapshot, read_json, read_jsonl, setup_runtime
setup_runtime()
from door8.data import uniform_sample
from door8.geometry import fit_scale, classify_measurement, depth_measure
from door8.metrics import scores, aggregate_sessions, paired_session_comparison


class Safeguards(unittest.TestCase):
    def test_output_escape(self):
        with self.assertRaises(ValueError):outpath('../outside.txt')
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p=Path(tmp)/'escape';p.symlink_to('/tmp',target_is_directory=True)
            with self.assertRaises(ValueError):outpath(p/'not_allowed.txt')

    def test_group_overlap_rejected(self):
        splits={'partitions':{'0':{'train':['a'],'val':['a'],'test':['b']}}}
        with self.assertRaisesRegex(ValueError,'leakage'):partition([],splits,'0')

    def test_exact_image_leakage_rejected(self):
        rows=[{'group_id':'a','rgb_sha256':'same'},{'group_id':'b','rgb_sha256':'same'}]
        splits={'partitions':{'0':{'train':['a'],'val':[],'test':['b']}}}
        with self.assertRaisesRegex(ValueError,'Identical RGB'):partition(rows,splits,'0')

    def test_actual_folds_and_class_scope(self):
        sp,summary,rows,splits=snapshot();cfg=read_json(sp/'protocol.json')
        self.assertEqual(len(cfg['classes']),8);self.assertFalse(any('E23' in r['class_name'] for r in rows))
        seen=[]
        for f in range(5):
            parts=partition(rows,splits,str(f));seen += [r['frame_id'] for r in parts['test']]
            for p in parts.values():self.assertEqual({r['class_name'] for r in p},set(cfg['classes']))
        self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),{r['frame_id'] for r in rows})
        temporal=partition(rows,splits,'temporal')
        self.assertTrue(all(r['session_dir'][:8]<'20261003' for r in temporal['train']+temporal['val']))
        self.assertTrue(all(r['session_dir'][:8]>='20261003' for r in temporal['test']))

    def test_lab_manual_only_and_disjoint(self):
        p,_,rows,_=snapshot();a=read_jsonl(p/'annotations_lab.jsonl');s=read_json(p/'annotations_lab_split.json')
        self.assertEqual(len(a),118);self.assertTrue(all(r['source']=='human_manual_lab' for r in a))
        self.assertFalse(set(s['train'])&set(s['val']))
        self.assertFalse({r['rgb_sha256'] for r in a}&{r['rgb_sha256'] for r in rows})

    def test_abstentions_remain_in_denominator(self):
        m=scores([{'class_name':'A','pred':'A'},{'class_name':'A','pred':None},{'class_name':'B','pred':'B'}],['A','B'])
        self.assertAlmostEqual(m['accuracy_all'],2/3);self.assertEqual(m['accuracy_judged'],1.)
        self.assertEqual(m['per_class']['A']['recall'],.5)

    def test_session_tie_abstains(self):
        rows=[{'session_dir':'x','group_id':'x','class_name':'A','pred':p} for p in ['A','B',None]]
        self.assertIsNone(aggregate_sessions(rows)[0]['pred'])

    def test_pairing_requires_identical_inputs(self):
        with self.assertRaises(ValueError):paired_session_comparison([{'frame_id':'a'}],[{'frame_id':'b'}],['A'])

    def test_scale_is_train_only_and_session_balanced(self):
        train=[{'frame_id':str(i),'session_dir':'long' if i<9 else 'short','class_name':'A','group_id':str(i),
                'intrinsics':{'serial':1}} for i in range(10)]
        feats={str(i):{'gate':'ok','span_normalized':1. if i<9 else 2.} for i in range(10)}
        feats['test_leak']={'gate':'ok','span_normalized':.001}
        cal=fit_scale(train,feats,{'A':100},'pixel');self.assertEqual(cal['camera_scales']['1'],75.)
        self.assertNotIn('test_leak',cal['train_frame_ids'])
        cfg={'geometry':{'valid_distance_mm':[1,200],'unknown_distance_mm':30,'pose_z_mm':[1400,1500],'pose_tilt_deg':15},
             'classes':['A'],'cad_hole_distance_mm':[100]}
        pred,_,why=classify_measurement({'intrinsics':{'serial':2}},feats['0'],cal,cfg)
        self.assertIsNone(pred);self.assertEqual(why,'uncalibrated_camera')

    def test_uniform_sampling_is_bounded(self):
        self.assertEqual(uniform_sample(list(range(100)),3),[0,50,99])
        self.assertEqual(uniform_sample([1,2],20),[1,2])

    def test_depth_plane_with_scored_landmarks(self):
        import numpy as np
        depth=np.full((200,300),1000,np.uint16)
        pts=[[30,30,.9],[130,30,.9],[30,130,.9],[130,130,.9],[20,80,.9],[240,80,.9]]
        result=depth_measure(depth,pts,pts[-2],pts[-1],{'fx':1000,'fy':1000,'cx':150,'cy':100})
        self.assertAlmostEqual(result['depth_distance_raw_mm'],220,places=5)
        self.assertAlmostEqual(result['plane_z_mm'],1000,places=5)

    def test_pilot_weights_are_rejected(self):
        from door8.learning import load_run
        if not (ROOT/'runs/pilot_rgb_f0_s42/run.json').exists():self.skipTest('Requires local pilot integration artifact')
        p,_,_,_=snapshot()
        with self.assertRaisesRegex(ValueError,'Pilot weights'):
            load_run('pilot_rgb_f0_s42',p)

    def test_pilot_features_are_rejected(self):
        from door8.features import load_features
        if not (ROOT/'features/pilot_holes_v2/features.json').exists():self.skipTest('Requires local pilot integration artifact')
        p,_,_,_=snapshot()
        with self.assertRaisesRegex(ValueError,'Pilot features'):
            load_features('pilot_holes_v2',p)


if __name__=='__main__':unittest.main()

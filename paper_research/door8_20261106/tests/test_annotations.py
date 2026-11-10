"""Use synthetic fixtures only; never create human labels for research images."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from door8.annotations import AnnotationStore, KEYS, RevisionConflict
from door8.core import ROOT, jsonl, read_json, sha256
sys.path.insert(0, str(ROOT / 'scripts'))
from annotate_field_web import make_server


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'cache').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='annotation_test_', dir=ROOT / 'cache')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'synthetic.png'
        Image.new('RGB', (80, 60), (80, 120, 160)).save(self.image)
        self.row = dict(annotation_id='synthetic_test', rgb=str(self.image),
                        rgb_sha256=sha256(self.image), class_name='TEST_ONLY',
                        session_dir='synthetic_only', cv_group=0,
                        roles_by_fold={'0': 'test'},
                        points={k: None for k in KEYS}, visible={k: True for k in KEYS},
                        annotator='', annotation_status='unannotated')
        queue = self.root / 'queue.jsonl'
        jsonl(queue, [self.row])
        self.store = AnnotationStore(queue, self.root / 'labels')
        self.payload = dict(annotation_id='synthetic_test', annotator='TEST_ONLY',
                            points={k: [10+i*5, 20] for i, k in enumerate(KEYS)},
                            visible={k: True for k in KEYS}, revision=None)

    def test_incomplete_labels_do_not_write(self):
        self.payload['points'][KEYS[0]] = None
        with self.assertRaises(ValueError):
            self.store.save(self.payload)
        self.assertFalse(self.store.labeldir.exists())

    def test_coordinates_are_finite_inside_original_and_visibility_is_consistent(self):
        for point in ([80, 20], [-1, 20], [20, 60], [float('nan'), 1], [True, 2], [1]):
            with self.subTest(point=point):
                payload = deepcopy(self.payload)
                payload['points'][KEYS[0]] = point
                with self.assertRaises(ValueError):
                    self.store.save(payload)
        payload = deepcopy(self.payload)
        payload['visible'][KEYS[0]] = False
        with self.assertRaises(ValueError):
            self.store.save(payload)
        payload['points'][KEYS[0]] = None
        self.store.save(payload)

    def test_save_preserves_metadata_and_rejects_stale_overwrite(self):
        result = self.store.save(self.payload)
        path = self.store.path('synthetic_test')
        first = read_json(path)
        self.assertEqual(first['roles_by_fold'], self.row['roles_by_fold'])
        self.assertEqual(first['rgb_sha256'], self.row['rgb_sha256'])
        self.assertEqual(first['source'], 'human_manual_field')
        with self.assertRaises(RevisionConflict):
            self.store.save(self.payload)
        self.payload['revision'] = result['revision']
        self.payload['points'][KEYS[0]] = [12, 22]
        self.store.save(self.payload)
        history = list((self.store.labeldir / 'history').glob('*.json'))
        self.assertEqual(len(history), 1)
        self.assertEqual(read_json(history[0]), first)
        self.assertEqual(self.store.item('synthetic_test')['points'][KEYS[0]], [12, 22])

    def test_client_cannot_change_paths_class_or_splits(self):
        for key in ('rgb', 'class_name', 'roles_by_fold'):
            payload = deepcopy(self.payload)
            payload[key] = 'not_allowed'
            with self.assertRaises(ValueError):
                self.store.save(payload)
        payload = deepcopy(self.payload)
        payload['annotation_id'] = '../../escape'
        with self.assertRaises(ValueError):
            self.store.save(payload)

    def test_changed_source_is_rejected(self):
        Image.new('RGB', (80, 60), 'red').save(self.image)
        with self.assertRaises(ValueError):
            self.store.save(self.payload)
        with self.assertRaises(ValueError):
            self.store.item('synthetic_test')

    def test_http_load_save_and_access_checks(self):
        server = make_server(self.store, 0, token='test-token')
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def request(path, payload=None, **headers):
            data = None if payload is None else json.dumps(payload).encode()
            return urlopen(Request(base + path, data=data, headers=headers), timeout=5)

        try:
            with request('/') as response:
                html = response.read().decode()
                self.assertIn('현장 홀 좌표', html)
                self.assertIn('name="annotation-token" content="test-token"', html)
            with request('/boot.js') as response:
                self.assertIn(b'door8StartupError', response.read())
            with self.assertRaises(HTTPError) as caught:
                request('/api/items')
            self.assertEqual(caught.exception.code, 403)
            auth = {'X-Annotation-Token': 'test-token'}
            with request('/api/items', **auth) as response:
                self.assertEqual(len(json.load(response)['items']), 1)
            with self.assertRaises(HTTPError) as caught:
                request('/api/items', **auth, Origin='http://unrelated.example')
            self.assertEqual(caught.exception.code, 403)
            with self.assertRaises(HTTPError) as caught:
                request('/api/items', **auth, Host='unrelated.example')
            self.assertEqual(caught.exception.code, 403)
            with request('/api/image?id=synthetic_test', **auth) as response:
                self.assertEqual(response.read(), self.image.read_bytes())
            with request('/api/save', self.payload, **auth, **{'Content-Type': 'application/json'}) as response:
                self.assertTrue(json.load(response)['revision'])
            with request('/api/item?id=synthetic_test', **auth) as response:
                row = json.load(response)
                self.assertNotIn('rgb', row)
                self.assertEqual(row['points'], self.payload['points'])
            with self.assertRaises(HTTPError) as caught:
                request('/api/save', self.payload, **auth, **{'Content-Type': 'application/json'})
            self.assertEqual(caught.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()

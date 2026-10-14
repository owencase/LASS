import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.quiz import QuizStore, generate_exam, same_concept, review_concept
from src.storage import LectureRepository
from src.llm import ModelStatus

FACTS = '''스택은 마지막 자료부터 꺼내는 LIFO 구조이다.
큐는 먼저 넣은 자료부터 꺼내는 FIFO 구조이다.
이진 탐색은 정렬된 구간의 중앙값을 비교한다.
선형 탐색은 처음 원소부터 차례대로 검사한다.
배열은 원소를 연속된 메모리에 저장한다.
연결 리스트는 노드의 참조로 자료를 연결한다.
원자성은 모든 작업이 수행되거나 모두 취소됨을 뜻한다.
지속성은 완료된 트랜잭션 결과를 영구 보존한다.
기본 키는 테이블에서 행을 고유하게 식별한다.
외래 키는 다른 테이블의 키를 참조한다.
TCP는 신뢰성 있는 바이트 스트림을 제공한다.
UDP는 패킷 도착 순서를 보장하지 않는다.
HTTP는 애플리케이션 계층의 프로토콜이다.
DNS는 도메인 이름을 IP 주소로 변환한다.
프로세스는 실행 중인 프로그램이다.
스레드는 같은 프로세스의 주소 공간을 공유한다.
교착 상태는 서로 점유한 자원을 기다리는 상태이다.
상호 배제는 임계 영역의 동시 접근을 제한한다.
캐시는 자주 쓰는 자료를 빠르게 보관한다.
가상 메모리는 논리적인 주소 공간을 제공한다.
컴파일러는 소스 코드를 목적 코드로 번역한다.
운영체제는 프로세스와 입출력 자원을 관리한다.'''

class RecoveryTests(unittest.TestCase):
    def test_resume_after_restart_keeps_validated_questions_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / 'lass.db')
            folder = repo.create_folder('과목')
            store = QuizStore(repo)
            note = SimpleNamespace(id='n', title='강의', source_name='강의.pdf', transcript=FACTS, summary='', notes='')
            failing = True
            def respond(system, prompt, **kwargs):
                if failing and len(store.draft(folder.id)) == 3:
                    return '{}'
                focus = prompt.split('[출제 대상]\n',1)[1].split('\n[이전 오류]',1)[0]
                item = dict(question=focus, correct_answer='O' if '정답은 반드시 "O"' in prompt else 'X', evidence_quote=focus)
                if prompt.startswith('MULTIPLE_CHOICE'):
                    item.update(correct_answer='정답', distractors=['오답1','오답2','오답3'])
                return json.dumps(item, ensure_ascii=False)
            pipeline = SimpleNamespace(llm_client=SimpleNamespace(chat=Mock(side_effect=respond)), unload=Mock())
            with patch('src.quiz.review_concept', return_value=False), patch('src.quiz.verify_question', side_effect=lambda p,s,q,o: ('O' if len(store.draft(folder.id)) % 2 == 0 else 'X') if o == ['O','X'] else '정답'):
                with self.assertRaisesRegex(ValueError, '4번 문항'):
                    generate_exam(pipeline, [note], store=store, folder_id=folder.id, max_attempts=3, retry_delay=0)
                partial = store.draft(folder.id)
                self.assertEqual(len(partial),3)
                self.assertEqual(pipeline.llm_client.chat.call_count,6)
                failing = False
                store = QuizStore(LectureRepository(repo.database_path))
                generated = generate_exam(pipeline, [note], store=store, folder_id=folder.id, max_attempts=3, retry_delay=0)
                self.assertEqual(generated[:3],partial)
                self.assertEqual(len(store.concepts(folder.id)),20)
                quiz_id = store.finish(folder.id,generated)
                self.assertEqual(store.get(quiz_id)['questions'],generated)
                self.assertEqual(store.draft(folder.id),[])
                with self.assertRaisesRegex(ValueError,'개념이 부족'):
                    generate_exam(pipeline,[note],store=store,folder_id=folder.id, max_attempts=3, retry_delay=0)
                self.assertEqual(len(store.list(folder.id)),1)

    def test_concepts_ignore_question_format_and_review_synonyms(self):
        self.assertTrue(same_concept('스택은 마지막 자료부터 꺼낸다.', '스택의 크기는 자료 수이다.'))
        self.assertFalse(same_concept('HTTP는 애플리케이션 계층이다.', 'DNS는 이름을 주소로 변환한다.'))
        pipeline = SimpleNamespace(llm_client=SimpleNamespace(chat=Mock(return_value='{"duplicate":true}')))
        self.assertTrue(review_concept(pipeline,'후입선출 구조는 나중 자료부터 꺼낸다.',['스택은 LIFO이다.']))
        self.assertIn('스택은 LIFO이다.',pipeline.llm_client.chat.call_args.args[1])

    def test_concept_review_retries_only_malformed_verdict(self):
        client = Mock(side_effect=['{}', '{"schema":true}', '{"duplicate":false}'])
        pipeline = SimpleNamespace(llm_client=SimpleNamespace(chat=client))
        self.assertFalse(review_concept(pipeline, 'DNS는 이름을 주소로 변환한다.', ['스택은 LIFO이다.']))
        self.assertEqual(client.call_count, 3)
        pipeline.llm_client.chat = Mock(return_value='{}')
        with self.assertRaisesRegex(ValueError, '저장된 문항을 유지'):
            review_concept(pipeline, '새 개념', ['이전 개념'])
        self.assertEqual(pipeline.llm_client.chat.call_count, 6)

    def test_navigation_restores_both_panels_for_every_entry_state(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'LASS_DATA_DIR':directory, 'LASS_EXAONE_MODEL_DIR':str(Path(directory)/'no-model')}), patch('src.ui.app.LocalModelManager.status',return_value=ModelStatus('test',Path(directory),True,1,(),1,True)):
            app = AppTest.from_file(str(Path(__file__).parents[1]/'src/ui/app.py'),default_timeout=30).run()
            for mode, button in [('chat','chatting-button'),('quiz','quiz-button')]:
                for left,right in [(False,False),(False,True),(True,False),(True,True)]:
                    app.session_state['library_open']=left
                    app.session_state['right_panel_open']=right
                    app.run()
                    app.button(key=button).click().run()
                    self.assertEqual(app.session_state['notes-panel-snapshot'],dict(library_open=left,right_panel_open=right))
                    app.button(key=mode+'-close').click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state['library_open'],left)
                    self.assertEqual(app.session_state['right_panel_open'],right)

    def test_auto_healing_continues_past_three_failures_without_regenerating_good_items(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory) / 'lass.db')
            folder = repo.create_folder('자동복구')
            store = QuizStore(repo)
            note = SimpleNamespace(id='n', title='강의', source_name='강의.pdf', transcript=FACTS, summary='', notes='')
            failures = []
            checkpoints = []
            def respond(system, prompt, **kwargs):
                if len(store.draft(folder.id)) == 3 and len(failures) < 6:
                    failures.append(1)
                    if len(failures) == 2:
                        raise OSError('temporary worker interruption')
                    return '{}'
                focus = prompt.split('[출제 대상]\n',1)[1].split('\n[이전 오류]',1)[0]
                item = dict(question=focus, correct_answer='O' if '정답은 반드시 "O"' in prompt else 'X', evidence_quote=focus)
                if prompt.startswith('MULTIPLE_CHOICE'):
                    item.update(correct_answer='정답', distractors=['오답1','오답2','오답3'])
                return json.dumps(item, ensure_ascii=False)
            client = SimpleNamespace(chat=Mock(side_effect=respond), unload=Mock())
            pipeline = SimpleNamespace(llm_client=client, unload=Mock())
            original_checkpoint = store.checkpoint
            def checkpoint(folder_id, questions, focus):
                checkpoints.append(questions[-1]['question_number'])
                original_checkpoint(folder_id, questions, focus)
            progress = []
            with patch('src.quiz.review_concept', return_value=False), patch('src.quiz.verify_question', side_effect=lambda p,s,q,o: ('O' if len(store.draft(folder.id)) % 2 == 0 else 'X') if o == ['O','X'] else '정답'), patch.object(store, 'checkpoint', side_effect=checkpoint):
                generated = generate_exam(pipeline, [note], store=store, folder_id=folder.id,
                                          progress=progress.append, retry_delay=0)
            self.assertEqual(len(generated),20)
            self.assertEqual(len(failures),6)
            self.assertEqual(checkpoints, list(range(1,21)))
            self.assertTrue(any('자동 복구' in message for message in progress))
            client.unload.assert_called_once()

if __name__ == '__main__':
    unittest.main()

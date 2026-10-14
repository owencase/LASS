import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config import Settings
from src.pipeline import LASSPipeline
from src.quiz import ChoiceDraft, QuizStore, generate_exam, validate_learning_question, validate_exam, verify_question
from src.storage import LectureRepository


def exam():
    return [{"question_type": "OX" if number <= 10 else "MULTIPLE_CHOICE", "question_number": number,
        "question": f"노트에 제시된 학습 원리에 관한 {number}번 지문입니다.",
        "options": ["O", "X"] if number <= 10 else ["보기 1", "보기 2", "보기 3", "보기 4"],
        "correct_answer": "O" if number <= 10 else "보기 2", "source_reference": "강의.pdf - 3페이지"}
        for number in range(1, 21)]


class QuizTests(unittest.TestCase):
    def test_mechanical_source_matching_and_raw_fragments_are_rejected(self):
        for text in ('“servo.detach()”라는 설명은 원문의 내용과 일치한다.', '- 실행화면(로그인요청:login.html)', '원문에 등장하는 문장은 무엇인가요?'):
            with self.assertRaises(ValueError):
                validate_learning_question(text)
        validate_learning_question('스택에서 가장 나중에 넣은 원소가 먼저 제거된다.')

    def test_independent_verdict_rejects_ambiguous_questions(self):
        pipeline = SimpleNamespace(llm_client=Mock())
        source = '스택은 후입선출 구조이다.'
        pipeline.llm_client.chat.return_value = json.dumps(dict(answer='O', meaningful=True, evidence_quote=source))
        self.assertEqual(verify_question(pipeline, source, "평서문입니다.", ["O", "X"]), "O")
        pipeline.llm_client.chat.return_value = json.dumps(dict(unambiguous=True, correct_indices=[1,2], meaningful=True, evidence_quote=source))
        with self.assertRaises(ValueError):
            verify_question(pipeline, source, "질문", ["가", "나", "다", "라"])

    def test_choice_draft_accepts_separate_distractors_and_combined_choices(self):
        draft = {"question": "원문에 따른 올바른 설명은 무엇인가요?", "correct_answer": "정답",
            "evidence_quote": "원문의 충분한 근거 인용문", "options": ["오답1", "정답", "오답2", "오답3"]}
        self.assertEqual(ChoiceDraft.model_validate(draft).distractors, ["오답1", "오답2", "오답3"])
        draft["options"].remove("정답")
        self.assertEqual(len(ChoiceDraft.model_validate(draft).distractors), 3)

    def test_verdict_requires_original_evidence_and_learning_value(self):
        pipeline = SimpleNamespace(llm_client=Mock())
        source = '스택은 후입선출 구조이다.'
        for meaningful, quote in ((False, source), (True, '본문에 없는 다른 자료구조이다.')):
            pipeline.llm_client.chat.return_value = json.dumps(dict(answer='O', meaningful=meaningful, evidence_quote=quote))
            with self.assertRaises(ValueError):
                verify_question(pipeline, source, '스택은 나중에 넣은 자료를 먼저 제거한다.', ['O','X'])

    def test_writer_verifier_disagreement_is_never_checkpointed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = LectureRepository(Path(directory)/'library.db')
            folder = repo.create_folder('검수')
            store = QuizStore(repo)
            source = '스택은 마지막 자료부터 꺼내는 후입선출 구조이다.'
            note = SimpleNamespace(id='n', title='강의', source_name='', transcript=source, summary='', notes='')
            client = Mock()
            client.chat.return_value = json.dumps(dict(question=source, correct_answer='O', evidence_quote=source))
            pipeline = SimpleNamespace(llm_client=client, unload=Mock())
            with patch('src.quiz.review_concept', return_value=False), patch('src.quiz.verify_question', return_value='X'):
                with self.assertRaisesRegex(ValueError, '정답.*다릅니다'):
                    generate_exam(pipeline, [note], store=store, folder_id=folder.id, max_attempts=1, retry_delay=0)
            self.assertEqual(store.draft(folder.id), [])
            self.assertEqual(store.concepts(folder.id), [])

    def test_exam_validation_rejects_wrong_count_type_answer_and_duplicates(self):
        questions = exam()
        self.assertEqual(len(validate_exam(questions)), 20)
        invalid_cases = [questions[:-1]]
        for field, value in (("question", "외래 키의 정의는 무엇인가요?"), ("correct_answer", "5"), ("options", ["O", "O"]), ("question_type", "ESSAY"), ("question_number", 3)):
            invalid = copy.deepcopy(questions)
            invalid[0][field] = value
            invalid_cases.append(invalid)
        duplicate = copy.deepcopy(questions)
        duplicate[1]["question"] = duplicate[0]["question"]
        invalid_cases.append(duplicate)
        for invalid in invalid_cases:
            with self.assertRaises(ValueError):
                validate_exam(invalid)

    def test_saved_exam_retake_grading_idempotence_and_folder_cascade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lass.db"
            repository = LectureRepository(path)
            folder = repository.create_folder("과목")
            store = QuizStore(repository)
            quiz_id = store.save(folder.id, exam())
            reopened = QuizStore(LectureRepository(path))
            self.assertEqual(reopened.get(quiz_id)["questions"], exam())
            answers = [item["correct_answer"] for item in exam()]
            with self.assertRaises(ValueError):
                reopened.submit(quiz_id, answers[:-1], 2, "incomplete")
            with self.assertRaises(ValueError):
                reopened.submit(quiz_id, answers, float("nan"), "nan")
            first = reopened.submit(quiz_id, answers, 12, "first")
            self.assertEqual(first["score"], 100)
            self.assertEqual(reopened.submit(quiz_id, answers, 99, "first")["elapsed_seconds"], 12)
            answers[0] = "X"
            self.assertEqual(reopened.submit(quiz_id, answers, 20, "retake")["score"], 95)
            second_id = store.save(folder.id, exam())
            self.assertNotEqual(quiz_id, second_id)
            self.assertEqual(len(store.list(folder.id)), 2)
            self.assertEqual(len(store.attempts(folder.id)), 2)
            repository.delete_folder(folder.id)
            self.assertEqual(store.attempts(folder.id), [])
            self.assertEqual(store.list(folder.id), [])

    def test_generation_validates_original_quote_and_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LASSPipeline(Settings(data_dir=Path(directory)))
            note = SimpleNamespace(id="note", title="강의", source_name="강의.pdf", transcript="## 3페이지\n\n" + "\n".join(f"개념{number}은 서로 다른 고유한 자료구조{number}에 대한 설명이다." for number in range(30)), summary="", notes="")
            calls = 0
            def respond(system, prompt, **kwargs):
                nonlocal calls
                calls += 1
                focus = prompt.split("[출제 대상]\n", 1)[1].split("\n[이전 오류]", 1)[0]
                number = calls - 1 if calls > 1 else 1
                item = {"question": f"고유한 자료구조에 대한 {number}번 설명입니다.",
                    "correct_answer": "O" if number % 2 else "X", "evidence_quote": focus}
                if calls == 1:
                    item["evidence_quote"] = "원문에 없는 인용문은 거부됩니다."
                if number > 10:
                    item.update(correct_answer="보기 2", distractors=["보기 1", "보기 3", "보기 4"])
                return json.dumps(item, ensure_ascii=False)
            pipeline.llm_client.chat = Mock(side_effect=respond)
            # Subject keys enforce uniqueness even when the rest of a sentence is identical.
            with patch("src.quiz.review_concept", return_value=False), patch("src.quiz.verify_question", return_value="보기 2"), patch("src.quiz.same_concept", side_effect=lambda a,b: a.split("은")[0] == b.split("은")[0]):
                # The independent checker agrees with each writer answer, including O/X.
                with patch("src.quiz.verify_question", side_effect=lambda p,s,q,o: ('O' if int(q.split('대한 ')[1].split('번')[0]) % 2 else 'X') if o == ['O','X'] else '보기 2'):
                    generated = generate_exam(pipeline, [note], retry_delay=0)
            self.assertEqual(len(generated), 20)
            self.assertEqual(pipeline.llm_client.chat.call_count, 21)



if __name__ == "__main__":
    unittest.main()

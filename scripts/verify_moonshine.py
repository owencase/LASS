from pathlib import Path
import argparse,json,os,shutil,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--exe',required=True,type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
results={}
for lang,name in [('ko','ko-fleurs.wav'),('en','en-beckett.wav')]:
    work=root/'validation'/lang;work.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'tests/assets'/name,work/'test.wav')
    env=os.environ.copy()
    env.update(LASS_STT_LANGUAGE=lang,LASS_STT_MODEL_DIR=str(root/'models/moonshine'),
               LASS_EXAONE_MODEL_DIR=str(root/'models/exaone-4.0-1.2b-iq4-xs'))
    env.pop('PYTHONPATH',None)
    env['PATH']=env['WINDIR']+';'+env['WINDIR']+'/System32'
    process=subprocess.run([str(args.exe),'--root',str(work),'--self-test'],cwd=work,env=env,
                           timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    if process.returncode:
        raise RuntimeError((work/'runtime-error.log').read_text(encoding='utf-8'))
    text=(work/'transcription-test.txt').read_text(encoding='utf-8')
    llm=(work/'llm-test.txt').read_text(encoding='utf-8').strip()
    assert llm=='2',llm
    if lang=='ko': assert any('가' <= c <= '힣' for c in text),text
    else: assert 'fail' in text.lower(),text
    results[lang]={'exit_code':process.returncode,'text':text,'llm':llm}
    print(lang,ascii(text),flush=True)
(root/'verification-stt-exe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

"""Verify the frozen first-run download path without reusing STT assets."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,subprocess,argparse,sys
parser=argparse.ArgumentParser();parser.add_argument('--exe',required=True,type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.stt.moonshine import CATALOG
work=root/'validation'/('fresh-download-2.1.1-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
work.mkdir(parents=True)
models=work/'models/moonshine'
assert not models.exists()
env=os.environ.copy();env.pop('PYTHONPATH',None)
env['PATH']=env['WINDIR']+';'+env['WINDIR']+'/System32'
env['LASS_STT_MODEL_DIR']=str(models)
env['LASS_EXAONE_MODEL_DIR']=str(root/'models/exaone-4.0-1.2b-iq4-xs')
process=subprocess.run([str(args.exe),'--root',str(work),'--prepare-models'],cwd=work,env=env,
                       timeout=240,creationflags=subprocess.CREATE_NO_WINDOW)
if process.returncode:
    error=work/'runtime-error.log'
    raise RuntimeError(error.read_text(encoding='utf-8') if error.exists() else str(process.returncode))
verified=[]
for language,model in CATALOG.items():
    for file in model['files']:
        path=models/language/file['name']
        assert path.stat().st_size==file['size'],path
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        assert h.hexdigest()==file['sha256'],path
        verified.append(language+'/'+file['name'])
result={'exit_code':process.returncode,'reused_stt_cache':False,'verified_files':verified,
        'bytes':sum(f['size'] for m in CATALOG.values() for f in m['files']),'work_directory':str(work)}
(root/'verification-first-download-2.1.1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True),flush=True)

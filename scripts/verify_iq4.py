from pathlib import Path
import json,os,subprocess,time,argparse,sys
parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.config import Settings
s=Settings.from_env()
results=[]
for device in ('auto','cpu'):
    env=os.environ.copy()
    env['LASS_EXAONE_DEVICE']=device
    env['PYTHONIOENCODING']='utf-8'
    env['LASS_DATA_DIR']=str(root/'data')
    env.pop('PYTHONPATH',None)
    if args.exe:
        # Only Windows system paths: all inference DLLs must come from the bundle.
        env['PATH']=os.environ['WINDIR']+';'+os.environ['WINDIR']+'/System32'
        command=[str(args.exe),'--llm-worker']
    else:
        command=[sys.executable,'-m','src.inference_worker']
    started=time.perf_counter()
    with (root/f'worker-{device}.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            request={'config':{'model_dir':str(s.exaone_model_dir),'model_id':s.exaone_model_id,'device':device,'temperature':0,'max_input_tokens':2048},
                'messages':[{'role':'system','content':'주어진 자료에서 정답만 짧게 한국어로 답하세요.'},
                            {'role':'user','content':'자료: 수강생은 5명이고 강사는 2명이다. 질문: 수강생은 몇 명인가?'}]}
            p.stdin.write((json.dumps(request,ensure_ascii=True)+'\n').encode('ascii'));p.stdin.flush()
            tokens=[];active=None;completed=False
            for line in p.stdout:
                message=json.loads(line.decode('ascii'))
                if 'error' in message: raise RuntimeError(message['error'])
                if 'device' in message: active=message['device']
                if 'token' in message: tokens.append(message['token'])
                if message.get('done'): completed=True;break
            answer=''.join(tokens).strip()
            assert completed and '5' in answer,(device,answer)
            if device=='cpu': assert active=='cpu',active
            from src.llm.native_runtime import gpu_available
            if device=='auto' and gpu_available(): assert active=='cuda',active
            results.append({'requested':device,'active':active,'answer':answer,'seconds':round(time.perf_counter()-started,2),'chunks':len(tokens)})
        finally:
            p.stdin.close()
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill();p.wait();raise
            p.stdout.close()
    print(json.dumps(results[-1],ensure_ascii=True),flush=True)
path=root/('verification-exe.json' if args.exe else 'verification-source.json')
path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

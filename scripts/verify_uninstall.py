"""Exercise the real Inno uninstall rules against disposable fixture data."""
from pathlib import Path
import json,re,subprocess,tempfile,uuid,time
root=Path(__file__).resolve().parents[1]
validation=(root/'validation').resolve();validation.mkdir(exist_ok=True)
source=(root/'scripts/installer.iss').read_text(encoding='utf-8')
def section(name):
    match=re.search(r'(?ms)^\['+re.escape(name)+r'\]\n(.*?)(?=^\[|\Z)',source)
    assert match,name
    return '['+name+']\n'+match.group(1)
compiler=root/'tools/inno/app/ISCC.exe'
with tempfile.TemporaryDirectory(prefix='uninstall-test-',dir=validation) as temporary:
    work=Path(temporary).resolve()
    assert work.is_relative_to(validation)
    app=work/'LASS-IQ4-Moonshine';data=work/'owned-data';other=work/'LASS-IQ4'
    assert app.resolve().is_relative_to(work) and data.resolve().is_relative_to(work)
    fixture=work/'fixture';(fixture/'_internal/src').mkdir(parents=True)
    (fixture/'LASS.exe').write_bytes(b'uninstall fixture; never executed')
    (fixture/'_internal/src/__init__.py').write_text('__version__="2.1.1"')
    output=work/'output';output.mkdir()
    script=f'''#define MyDataRoot "{data}"
[Setup]
AppId={str(uuid.uuid4())}
AppName=LASS Uninstall Test
AppVersion=0.0.1
DefaultDirName={app}
OutputDir={output}
OutputBaseFilename=cleanup-test
PrivilegesRequired=lowest
CreateUninstallRegKey=no
UsePreviousAppDir=no
UninstallLogMode=overwrite
[Files]
Source: "{fixture}\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{root/'scripts/stop_installed_app.ps1'}"; DestDir: "{{app}}"
Source: "{root/'scripts/backup_install.ps1'}"; Flags: dontcopy
'''+section('UninstallRun')+section('UninstallDelete')+section('Code')
    iss=work/'cleanup-test.iss';iss.write_text(script,encoding='utf-8-sig')
    flags=dict(timeout=60,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    subprocess.run([str(compiler),str(iss)],stdout=subprocess.DEVNULL,**flags)
    subprocess.run([str(output/'cleanup-test.exe'),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-',f'/DIR={app}'],**flags)
    for relative in ['_internal/__pycache__/stale.pyc','.hidden-garbage/cache.bin']:
        path=app/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'leftover')
    for relative in ['data/notes.db','models/model.gguf','.cache/webview/old.bin','tmp/partial.bin']:
        path=data/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'owned data')
    other.mkdir();(other/'preserve.txt').write_text('other version')
    subprocess.run(['attrib','+h',str(app/'.hidden-garbage')],**flags)
    subprocess.run([str(app/'unins000.exe'),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART'],**flags)
    deadline=time.monotonic()+20
    while app.exists() and time.monotonic()<deadline:
        time.sleep(.1)
    assert not app.exists(),list(app.rglob('*'))
    assert not data.exists(),list(data.rglob('*'))
    assert (other/'preserve.txt').read_text()=='other version'
    result={'installation_removed':True,'hidden_garbage_removed':True,'notes_models_caches_removed':True,'other_version_preserved':True}
(root/'verification-uninstall-2.1.1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)

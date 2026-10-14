using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

// Keep the familiar dist\LASS.exe entry point without extracting the ML runtime.
class Launcher {
    static string Quote(string value) {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in value) {
            if (ch == '\\') { slashes++; continue; }
            result.Append('\\', ch == '"' ? slashes * 2 + 1 : slashes);
            result.Append(ch);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }
    [STAThread]
    static int Main(string[] args) {
        try {
            string exe = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "LASS", "LASS.exe");
            if (!File.Exists(exe)) throw new FileNotFoundException("LASS 폴더와 _internal을 실행 파일 옆에 함께 두세요.", exe);
            using (var process = Process.Start(new ProcessStartInfo(exe, String.Join(" ", Array.ConvertAll(args, Quote))) {
                UseShellExecute = false, CreateNoWindow = true
            })) {
                process.WaitForExit();
                return process.ExitCode;
            }
        } catch (Exception error) {
            MessageBox.Show(error.Message, "LASS", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}

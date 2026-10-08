using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Web.Script.Serialization;
using System.Collections.Generic;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, "runtime", "python", "python.exe");
        string app = Path.Combine(root, "runtime", "ui", "app.py");
        string session = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LivingHome", "Session", "connection.json");
        try
        {
            if (File.Exists(session))
            {
                try
                {
                    var saved = new JavaScriptSerializer().Deserialize<Dictionary<string, string>>(File.ReadAllText(session));
                    Uri origin = new Uri(saved["url"]);
                    if (origin.Scheme != "http" || origin.Host != "127.0.0.1" || origin.AbsolutePath != "/" || origin.Query != "" || origin.UserInfo != "") throw new Exception();
                    var request = (HttpWebRequest)WebRequest.Create(new Uri(origin, "api/status"));
                    request.Timeout = 1500; request.AllowAutoRedirect = false;
                    request.Headers["Authorization"] = "Bearer " + saved["session"];
                    using (var response = request.GetResponse()) { }
                    Process.Start(new ProcessStartInfo(origin.AbsoluteUri + "#" + Uri.EscapeDataString(saved["session"])) { UseShellExecute = true });
                    return 0;
                }
                catch { /* A stale session does not authorize terminating another process. */ }
            }
            if (!File.Exists(python) || !File.Exists(app))
                throw new Exception("Extract the complete Living Home archive before opening LivingHome.exe. Keep it beside its runtime folder.");
            var start = new ProcessStartInfo(python, Quote(app) + " --connection-file " + Quote(session))
            {
                WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden, RedirectStandardError = true
            };
            using (var child = Process.Start(start))
            {
                // Drain private error output without displaying tokens or household paths.
                child.ErrorDataReceived += (sender, args) => { };
                child.BeginErrorReadLine();
                child.WaitForExit();
                if (child.ExitCode != 0)
                    throw new Exception("Living Home could not start. Another app may be using ports 18880 or 18881, or the saved household configuration may need attention. Your data has been preserved. See START-HERE.md for recovery steps.");
            }
            return 0;
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "Living Home could not open", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 1;
        }
    }
    static string Quote(string value) { return "\"" + value.Replace("\"", "\\\"") + "\""; }
}

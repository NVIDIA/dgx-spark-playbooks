using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

[assembly: AssemblyTitle("Living Home Setup")]
[assembly: AssemblyDescription("Verified online/bootstrap installation for Living Home on Windows ARM64")]
[assembly: AssemblyCompany("Living Home")]
[assembly: AssemblyProduct("Living Home Setup")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

namespace LivingHome.Setup
{
    internal static class Program
    {
        [DllImport("kernel32.dll")] private static extern bool AttachConsole(int id);
        [STAThread]
        private static int Main(string[] args)
        {
            if (args.Length == 2 && args[0] == "--render-preview")
            {
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                try
                {
                    string output = Path.GetFullPath(args[1]);
                    Safety.NoReparse(output);
                    using (var window = new SetupWindow())
                    {
                        window.PreparePreview();
                        using (var bitmap = new Bitmap(window.Width, window.Height))
                        {
                            window.RenderPreview(bitmap);
                            bitmap.Save(output, System.Drawing.Imaging.ImageFormat.Png);
                        }
                    }
                    return 0;
                }
                catch { return 2; }
            }
            if (args.Length > 0)
            {
                AttachConsole(-1);
                return CommandLine.Run(args);
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SetupWindow());
            return 0;
        }
    }
    internal static class CommandLine
    {
        public static int Run(string[] args)
        {
            var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var flags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string reportPath = null;
            CancellationTokenSource stop = new CancellationTokenSource();
            ConsoleCancelEventHandler handler = (sender, e) => { e.Cancel = true; stop.Cancel(); };
            Console.CancelKeyPress += handler;
            try
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string arg = args[i];
                    if (arg == "--help" || arg == "-h") { Console.WriteLine(Help); return 0; }
                    if (arg == "--check-only" || arg == "--unattended")
                    {
                        if (!flags.Add(arg)) throw new SetupException("Duplicate argument: " + arg);
                    }
                    else if (arg == "--manifest" || arg == "--manifest-sha256" || arg == "--install-dir" || arg == "--report")
                    {
                        if (options.ContainsKey(arg) || ++i >= args.Length) throw new SetupException("Missing or duplicate value for " + arg);
                        options.Add(arg, args[i]);
                    }
                    else throw new SetupException("Unknown argument: " + arg + ". Use --help for usage.");
                }
                if (!flags.Contains("--check-only") && !flags.Contains("--unattended")) throw new SetupException("Command-line operations require --check-only or --unattended.");
                if (!options.ContainsKey("--manifest") || !options.ContainsKey("--install-dir")) throw new SetupException("Supply --manifest and --install-dir. No default online release is bundled.");
                if (options.ContainsKey("--report")) reportPath = Path.GetFullPath(options["--report"]);
                string pin = options.ContainsKey("--manifest-sha256") ? options["--manifest-sha256"] : null;
                if (flags.Contains("--unattended") && !flags.Contains("--check-only") && String.IsNullOrWhiteSpace(pin)) throw new SetupException("Unattended installation requires --manifest-sha256 from a trusted release publisher.");
                LoadedRelease release = ManifestReader.Read(options["--manifest"], pin, stop.Token);
                ReadinessReport readiness = Readiness.Check(release, options["--install-dir"], InstallerEngine.ReleaseCache(release));
                Console.WriteLine("Living Home release " + release.Manifest.releaseVersion + " | manifest SHA256 " + release.ManifestHash);
                foreach (CheckResult check in readiness.checks) Console.WriteLine(check.status.ToUpperInvariant() + " | " + check.name + " | " + check.detail);
                if (reportPath != null) SaveReport(reportPath, readiness);
                if (!readiness.canInstall) return 3;
                if (flags.Contains("--check-only")) { Console.WriteLine("Readiness check complete. No installation or asset download was performed."); return 0; }
                InstallResult result = InstallerEngine.Install(release, options["--install-dir"], stop.Token, update => Console.WriteLine(update.Message + (update.Total > 0 ? " | " + update.Percent + "%" : "")));
                if (reportPath != null) SaveReport(reportPath, new { status = "setup-complete", installDirectory = result.InstallDirectory, setupExitCode = result.SetupExitCode, readiness = readiness });
                Console.WriteLine("Installation complete: " + result.InstallDirectory);
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled. Verified and partial downloads are retained for resume.");
                if (reportPath != null) TrySaveError(reportPath, "cancelled", "Cancelled; cache retained for resume.");
                return 4;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("Setup failed: " + ex.Message);
                if (reportPath != null) TrySaveError(reportPath, "error", ex.Message);
                return 2;
            }
            finally { Console.CancelKeyPress -= handler; stop.Dispose(); }
        }
        private static void TrySaveError(string path, string status, string message)
        {
            try { SaveReport(path, new { status = status, message = message, recordedAtUtc = DateTime.UtcNow.ToString("o") }); }
            catch (Exception ex) { Console.Error.WriteLine("The report could not be saved: " + ex.Message); }
        }
        internal static void SaveReport(string path, object report)
        {
            Safety.NoReparse(path);
            File.WriteAllText(path, new JavaScriptSerializer().Serialize(report), new UTF8Encoding(false));
        }
        private const string Help =
            "Living Home Setup 1.0\r\n\r\n" +
            "Run without arguments to open the installer. No release hosting is preconfigured.\r\n\r\n" +
            "LivingHomeSetup.exe --check-only --manifest <file-or-HTTPS-URL> --install-dir <new-directory> [--manifest-sha256 <pin>] [--report <json-file>]\r\n" +
            "LivingHomeSetup.exe --unattended --manifest <file-or-HTTPS-URL> --manifest-sha256 <pin> --install-dir <new-directory> [--report <json-file>]\r\n\r\n" +
            "Check-only reads the manifest and checks hardware/storage, without downloading assets, creating the install directory, or launching setup.\r\n" +
            "Unattended setup verifies exact sizes and SHA256, extracts to a new directory, and waits for the release entry point. The entry point may have its own prompts.\r\n" +
            "Exit codes: 0 success; 2 invalid input/download/setup error; 3 readiness failed; 4 cancelled.\r\n" +
            "Use a trusted publisher's independently obtained manifest SHA256 pin; embedded payload hashes alone do not establish publisher identity.\r\n";
    }

    internal sealed class SetupWindow : Form
    {
        private readonly TextBox manifest = new TextBox();
        private readonly TextBox pin = new TextBox();
        private readonly TextBox directory = new TextBox();
        private readonly Button manifestBrowse = new Button();
        private readonly Button directoryBrowse = new Button();
        private readonly Button check = new Button();
        private readonly Button install = new Button();
        private readonly Button cancel = new Button();
        private readonly CheckBox trust = new CheckBox();
        private readonly Label releaseInfo = new Label();
        private readonly Label status = new Label();
        private readonly ProgressBar bar = new ProgressBar();
        private readonly ListView results = new ListView();
        private readonly TextBox activity = new TextBox();
        private LoadedRelease checkedRelease;
        private ReadinessReport checkedReport;
        private CancellationTokenSource current;
        private bool running;
        private bool setupRunning;
        private readonly List<string> log = new List<string>();

        public SetupWindow()
        {
            Text = "Living Home Setup";
            Font = new Font("Segoe UI", 10F);
            ClientSize = new Size(870, 760);
            MinimumSize = new Size(886, 799);
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Color.FromArgb(248, 250, 252);
            AutoScaleMode = AutoScaleMode.Font;
            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(28, 22, 28, 20), ColumnCount = 3, RowCount = 16 };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 160));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 125));
            foreach (int height in new[] { 46, 42, 32, 38, 38, 38, 54, 34, 150, 32, 26, 24, 80, 12, 44, 28 }) layout.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
            Controls.Add(layout);
            var title = new Label { Text = "Set up Living Home", Font = new Font("Segoe UI", 22F, FontStyle.Bold), ForeColor = Color.FromArgb(24, 37, 55), Dock = DockStyle.Fill };
            AddSpan(layout, title, 0);
            AddSpan(layout, new Label { Text = "For Windows ARM64 with NVIDIA • choose a trusted release, check readiness, then install.", Dock = DockStyle.Fill }, 1);
            AddSpan(layout, new Label { Text = "1   Choose the release and a new installation directory", Font = new Font(Font, FontStyle.Bold), Dock = DockStyle.Fill }, 2);
            manifest.PlaceholderTextCompat();
            AddInput(layout, "Release manifest", manifest, manifestBrowse, "Choose file…", 3);
            AddInput(layout, "SHA256 pin (optional)", pin, null, null, 4);
            directory.Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "LivingHome");
            AddInput(layout, "Install directory", directory, directoryBrowse, "Choose parent…", 5);
            releaseInfo.Text = "Enter a manifest file or HTTPS URL from the release publisher. No online release is bundled.";
            releaseInfo.Dock = DockStyle.Fill; releaseInfo.ForeColor = Color.FromArgb(74, 85, 104);
            AddSpan(layout, releaseInfo, 6);
            AddSpan(layout, new Label { Text = "2   Check this device", Font = new Font(Font, FontStyle.Bold), Dock = DockStyle.Fill }, 7);
            results.Dock = DockStyle.Fill; results.View = View.Details; results.FullRowSelect = true; results.GridLines = false;
            results.Columns.Add("Check", 155); results.Columns.Add("Result", 65); results.Columns.Add("Details", 530);
            AddSpan(layout, results, 8);
            trust.Text = "I trust the displayed publisher and release setup code."; trust.Dock = DockStyle.Fill; trust.Enabled = false;
            AddSpan(layout, trust, 9);
            status.Text = "Ready to check. Existing installations are protected."; status.Dock = DockStyle.Fill;
            AddSpan(layout, status, 10);
            bar.Dock = DockStyle.Fill; bar.Margin = new Padding(3, 3, 3, 5);
            AddSpan(layout, bar, 11);
            activity.Dock = DockStyle.Fill; activity.Multiline = true; activity.ScrollBars = ScrollBars.Vertical; activity.ReadOnly = true; activity.BackColor = Color.White;
            AddSpan(layout, activity, 12);
            var buttons = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.RightToLeft, WrapContents = false };
            cancel.Text = "Cancel"; cancel.Width = 105; cancel.Height = 34; cancel.Enabled = false;
            install.Text = "Install release"; install.Width = 145; install.Height = 34; install.Enabled = false;
            install.BackColor = Color.FromArgb(35, 96, 178); install.ForeColor = Color.White; install.FlatStyle = FlatStyle.Flat;
            check.Text = "Check readiness"; check.Width = 155; check.Height = 34;
            var save = new Button { Text = "Save report…", Width = 125, Height = 34 };
            buttons.Controls.Add(cancel); buttons.Controls.Add(install); buttons.Controls.Add(check); buttons.Controls.Add(save);
            AddSpan(layout, buttons, 14);
            AddSpan(layout, new Label { Text = "Downloads resume after interruption. GPU, Defender, and device settings require your own configuration.", Font = new Font("Segoe UI", 9F), ForeColor = Color.FromArgb(84, 97, 111), Dock = DockStyle.Fill }, 15);
            manifestBrowse.Click += (s, e) => { using (var dialog = new OpenFileDialog { Filter = "Release manifests (*.json)|*.json|All files (*.*)|*.*", CheckFileExists = true }) if (dialog.ShowDialog(this) == DialogResult.OK) manifest.Text = dialog.FileName; };
            directoryBrowse.Click += (s, e) =>
            {
                using (var dialog = new FolderBrowserDialog { Description = "Choose an existing parent. Setup will suggest a new LivingHome directory inside it.", ShowNewFolderButton = false })
                    if (dialog.ShowDialog(this) == DialogResult.OK)
                    {
                        string suggestion = Path.Combine(dialog.SelectedPath, "LivingHome"); int suffix = 2;
                        while (Directory.Exists(suggestion) || File.Exists(suggestion)) suggestion = Path.Combine(dialog.SelectedPath, "LivingHome-" + suffix++);
                        directory.Text = suggestion;
                    }
            };
            manifest.TextChanged += InvalidateCheck; pin.TextChanged += InvalidateCheck; directory.TextChanged += InvalidateCheck;
            trust.CheckedChanged += (s, e) => RefreshButtons();
            check.Click += (s, e) => StartCheck(); install.Click += (s, e) => StartInstall();
            cancel.Click += (s, e) => { if (current != null && !setupRunning) { current.Cancel(); cancel.Enabled = false; status.Text = "Cancelling safely; partial downloads will be retained…"; } };
            save.Click += (s, e) =>
            {
                using (var dialog = new SaveFileDialog { Filter = "JSON report (*.json)|*.json", FileName = "LivingHomeSetup-report.json" })
                    if (dialog.ShowDialog(this) == DialogResult.OK)
                        try { CommandLine.SaveReport(dialog.FileName, new { readiness = checkedReport, activity = log.ToArray() }); }
                        catch (Exception ex) { MessageBox.Show(this, ex.Message, "Report could not be saved", MessageBoxButtons.OK, MessageBoxIcon.Error); }
            };
            FormClosing += (s, e) =>
            {
                if (running) { e.Cancel = true; status.Text = setupRunning ? "Release setup is running. Wait for it to finish before closing." : "Use Cancel and wait for the current operation to stop before closing."; }
            };
        }
        internal void PreparePreview()
        {
            IntPtr handle = Handle;
            CreateChildren(this);
            PerformLayout();
        }
        private static void CreateChildren(Control parent)
        {
            foreach (Control child in parent.Controls) { IntPtr handle = child.Handle; CreateChildren(child); child.PerformLayout(); }
        }
        internal void RenderPreview(Bitmap bitmap)
        {
            DrawToBitmap(bitmap, new Rectangle(Point.Empty, bitmap.Size));
            Point origin = PointToScreen(Point.Empty);
            Point windowOrigin = new Point(Left, Top);
            RenderChildren(this, bitmap, new Point(origin.X - windowOrigin.X, origin.Y - windowOrigin.Y));
        }
        private static void RenderChildren(Control parent, Bitmap bitmap, Point origin)
        {
            foreach (Control child in parent.Controls)
            {
                Point offset = new Point(origin.X + child.Left, origin.Y + child.Top);
                if (child.Width <= 0 || child.Height <= 0) continue;
                child.DrawToBitmap(bitmap, new Rectangle(offset, child.Size));
                RenderChildren(child, bitmap, offset);
            }
        }
        private void AddInput(TableLayoutPanel table, string label, TextBox input, Button browse, string text, int row)
        {
            table.Controls.Add(new Label { Text = label, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft }, 0, row);
            input.Dock = DockStyle.Fill; input.Margin = new Padding(3, 5, 8, 3); table.Controls.Add(input, 1, row);
            if (browse != null) { browse.Text = text; browse.Dock = DockStyle.Fill; browse.Margin = new Padding(3, 3, 3, 4); table.Controls.Add(browse, 2, row); }
            else table.SetColumnSpan(input, 2);
        }
        private static void AddSpan(TableLayoutPanel table, Control control, int row) { table.Controls.Add(control, 0, row); table.SetColumnSpan(control, 3); }
        private void InvalidateCheck(object sender, EventArgs args)
        {
            if (running) return;
            checkedRelease = null; checkedReport = null; trust.Checked = false; trust.Enabled = false; results.Items.Clear();
            releaseInfo.Text = "Check readiness to inspect this release and its source."; RefreshButtons();
        }
        private void RefreshButtons()
        {
            check.Enabled = !running; manifest.Enabled = !running; pin.Enabled = !running; directory.Enabled = !running;
            manifestBrowse.Enabled = !running; directoryBrowse.Enabled = !running;
            install.Enabled = !running && checkedReport != null && checkedReport.canInstall && trust.Checked;
            trust.Enabled = !running && checkedRelease != null;
            cancel.Enabled = running && !setupRunning && current != null && !current.IsCancellationRequested;
        }
        private void AddLog(string message)
        {
            string line = DateTime.Now.ToString("HH:mm:ss") + "  " + message;
            log.Add(line); activity.AppendText(line + Environment.NewLine);
        }
        private void StartCheck()
        {
            string source = manifest.Text.Trim(), hash = pin.Text.Trim(), target = directory.Text.Trim();
            checkedRelease = null; checkedReport = null; results.Items.Clear(); trust.Checked = false;
            BeginWork("Reading the release manifest…");
            Task.Run(() =>
            {
                try
                {
                    LoadedRelease loaded = ManifestReader.Read(source, hash, current.Token);
                    ReadinessReport report = Readiness.Check(loaded, target, InstallerEngine.ReleaseCache(loaded));
                    Post(() =>
                    {
                        checkedRelease = loaded; checkedReport = report;
                        string publisher = loaded.ManifestSource.IsLocal ? "Local manifest: " + loaded.ManifestSource.FilePath : "Manifest publisher: " + loaded.ManifestSource.Display;
                        string payloadOrigin = loaded.PayloadSource.IsLocal ? "local file" : loaded.PayloadSource.Display;
                        releaseInfo.Text = "Release " + loaded.Manifest.releaseVersion + " • " + publisher + Environment.NewLine + "Payload: " + payloadOrigin + " • " + loaded.Manifest.models.Count + " model(s) • " + (loaded.IsPinned ? "SHA256 pin verified" : "manifest unpinned");
                        foreach (CheckResult item in report.checks)
                        {
                            var row = new ListViewItem(item.name); row.SubItems.Add(item.status.ToUpperInvariant()); row.SubItems.Add(item.detail);
                            row.ForeColor = item.status == "fail" ? Color.FromArgb(163, 35, 35) : item.status == "warning" ? Color.FromArgb(135, 83, 9) : Color.FromArgb(34, 100, 65);
                            row.ToolTipText = item.detail; results.Items.Add(row);
                        }
                        results.ShowItemToolTips = true;
                        AddLog("Manifest SHA256: " + loaded.ManifestHash);
                        EndWork(report.canInstall ? "Readiness passed. Review the publisher and check the trust box to install." : "Readiness failed. Review the red checks and resolve them on the target device.");
                    });
                }
                catch (Exception ex) { Post(() => EndWithError(ex)); }
            });
        }
        private void StartInstall()
        {
            if (checkedRelease == null || checkedReport == null || !checkedReport.canInstall || !trust.Checked) return;
            string source = manifest.Text.Trim(), pinText = pin.Text.Trim(), target = directory.Text.Trim(), expectedHash = checkedRelease.ManifestHash;
            BeginWork("Verifying the checked release again…");
            Task.Run(() =>
            {
                try
                {
                    LoadedRelease loaded = ManifestReader.Read(source, pinText, current.Token);
                    if (loaded.ManifestHash != expectedHash) throw new SetupException("The manifest changed after readiness checking. Check the new release and review its publisher again.");
                    string previous = null;
                    InstallResult installed = InstallerEngine.Install(loaded, target, current.Token, update => Post(() =>
                    {
                        status.Text = update.Message + (update.Total > 0 ? " — " + update.Percent + "%" : ""); bar.Value = update.Percent;
                        if (update.Message.StartsWith("Launching the release setup")) { setupRunning = true; cancel.Enabled = false; }
                        if (previous != update.Message) { AddLog(update.Message); previous = update.Message; }
                    }));
                    Post(() =>
                    {
                        checkedReport.canInstall = false;
                        EndWork("Installation complete: " + installed.InstallDirectory); bar.Value = 100;
                        MessageBox.Show(this, "Living Home release setup finished successfully.\r\n\r\n" + installed.InstallDirectory + "\r\n\r\nFollow the release's first-run instructions to validate the device and configure your household.", "Setup complete", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    });
                }
                catch (Exception ex) { Post(() => EndWithError(ex)); }
            });
        }
        private void BeginWork(string message) { running = true; setupRunning = false; current = new CancellationTokenSource(); bar.Value = 0; status.Text = message; AddLog(message); RefreshButtons(); }
        private void EndWork(string message) { running = false; setupRunning = false; if (current != null) { current.Dispose(); current = null; } status.Text = message; AddLog(message); RefreshButtons(); }
        private void EndWithError(Exception error)
        {
            bool cancelled = error is OperationCanceledException;
            EndWork(cancelled ? "Cancelled. Retry to resume saved downloads." : "Setup needs attention. " + error.Message);
            if (!cancelled) MessageBox.Show(this, error.Message, "Setup needs attention", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        private void Post(Action action) { if (!IsDisposed && IsHandleCreated) BeginInvoke(action); }
    }
    internal static class TextBoxCompat
    {
        // No placeholder API exists in the installed .NET Framework WinForms runtime.
        public static void PlaceholderTextCompat(this TextBox box) { box.AccessibleDescription = "Select a release manifest JSON file or enter an HTTPS URL."; }
    }
}

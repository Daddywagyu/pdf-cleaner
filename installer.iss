; Inno Setup script - Local PDF Pen Remover
#define MyAppName "Local PDF Pen Remover"
#define MyAppVersion "1.0.0"
#define MyAppExeName "LocalPDFPenRemover.exe"

[Setup]
AppId={{B7C1E2A4-5D3F-4B8E-9A21-6F0C4D7E1A55}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\LocalPDFPenRemover
DefaultGroupName={#MyAppName}
OutputDir=Output
OutputBaseFilename=LocalPDFPenRemover-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\LocalPDFPenRemover\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: postinstall nowait skipifsilent

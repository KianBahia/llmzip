"""Generate a Finder Quick Action bundle (NAME.workflow) that runs one shell command.

Used by finder/install.sh. Mirrors the layout of macOS's own Services workflows and of
Quick Actions saved by Automator: Contents/Info.plist (NSServices entry, Finder files)
plus Contents/document.wflow with a single "Run Shell Script" action (zsh, input passed
as arguments).

    python make_workflow.py --dest DIR --name "Compress with llmzip" --command '...'
"""

import argparse
import plistlib
import shutil
import uuid
from pathlib import Path

ACTION_PATH = "/System/Library/Automator/Run Shell Script.action"


def info_plist(name: str) -> dict:
    return {
        "CFBundleDevelopmentRegion": "en_US",
        "CFBundleIdentifier": "com.llmzip.quickaction." + name.lower().replace(" ", "-"),
        "CFBundleName": name,
        "CFBundleShortVersionString": "1.0",
        "NSServices": [{
            "NSMenuItem": {"default": name},
            "NSMessage": "runWorkflowAsService",
            "NSRequiredContext": {"NSApplicationIdentifier": "com.apple.finder"},
            "NSSendFileTypes": ["public.item"],
        }],
    }


def document_wflow(command: str) -> dict:
    ids = {k: str(uuid.uuid4()).upper() for k in ("input", "output", "action")}
    params = {
        "COMMAND_STRING": command,
        "CheckedForUserDefaultShell": True,
        "inputMethod": 1,  # pass input "as arguments"
        "shell": "/bin/zsh",
        "source": "",
    }
    action = {
        "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.string"]},
        "AMActionVersion": "2.0.3",
        "AMApplication": ["Automator"],
        "AMParameterProperties": {k: {} for k in params},
        "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
        "ActionBundlePath": ACTION_PATH,
        "ActionName": "Run Shell Script",
        "ActionParameters": params,
        "BundleIdentifier": "com.apple.RunShellScript",
        "CFBundleVersion": "2.0.3",
        "CanShowSelectedItemsWhenRun": False,
        "CanShowWhenRun": True,
        "Category": ["AMCategoryUtilities"],
        "Class Name": "RunShellScriptAction",
        "InputUUID": ids["input"],
        "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
        "OutputUUID": ids["output"],
        "UUID": ids["action"],
        "UnlocalizedApplications": ["Automator"],
        "arguments": {},
        "isViewVisible": 1,
        "location": "309.000000:253.000000",
        "nibPath": f"{ACTION_PATH}/Contents/Resources/Base.lproj/main.nib",
    }
    return {
        "AMApplicationBuild": "528",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [{"action": action, "isViewVisible": 1}],
        "connectors": {},
        "workflowMetaData": {
            "applicationBundleIDsByPath": {},
            "applicationPaths": [],
            "inputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "outputTypeIdentifier": "com.apple.Automator.nothing",
            "presentationMode": 15,
            "processesInput": 0,
            "serviceApplicationBundleID": "com.apple.finder",
            "serviceApplicationPath": "/System/Library/CoreServices/Finder.app",
            "serviceInputTypeIdentifier": "com.apple.Automator.fileSystemObject",
            "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceProcessesInput": 0,
            "systemImageName": "NSActionTemplate",
            "useAutomaticInputType": 0,
            "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
        },
    }


def write_workflow(dest: Path, name: str, command: str) -> Path:
    bundle = dest / f"{name}.workflow"
    if bundle.exists():
        shutil.rmtree(bundle)
    contents = bundle / "Contents"
    contents.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(info_plist(name), f)
    with (contents / "document.wflow").open("wb") as f:
        plistlib.dump(document_wflow(command), f)
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--command", required=True)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    print(write_workflow(args.dest, args.name, args.command))

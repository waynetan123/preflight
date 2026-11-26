# Preflight as an MCP server

`preflight serve --mcp` gives an MCP client the tools of the guarded robot: `describe_scene`, `move_to`, `pick` and `place`. The robot is the simulated arm in the kitchen scene. Before a call runs, the firewall simulates it and checks the result against the rules file. Only a call that passes moves the robot.

## Install

```sh
pip install 'preflight-robot[mcp]'
```

Preflight is not on PyPI yet. Until the release, install it from a clone of this repository:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[mcp]'
```

## Start the server

The MCP client starts the server and talks to it over stdin and stdout. Do not start the server in a terminal, except for a test.

```sh
preflight serve --mcp --rules scenes/kitchen_rules.yaml --audit audit.jsonl
```

| Option | Meaning |
| --- | --- |
| `--rules FILE` | The rules file. It names the scene. Necessary. |
| `--feedback detailed` | A blocked call gives the rule, the measured value, the limit and the time. This is the default. |
| `--feedback minimal` | A blocked call gives only "blocked by safety policy". Use it when the user can be an attacker. |
| `--audit FILE` | The server adds one JSON line to this file for each decision. |

## Add the server to Claude Code

Use absolute paths, so that the command works from any working directory. If Preflight is in a virtual environment, give the full path to its `preflight` command.

```sh
claude mcp add preflight -- /path/to/preflight/.venv/bin/preflight serve --mcp \
    --rules /path/to/preflight/scenes/kitchen_rules.yaml --audit /path/to/preflight/audit.jsonl
```

To see the status of the server, run `claude mcp list`.

## Add the server to Claude Desktop

On macOS, open `~/Library/Application Support/Claude/claude_desktop_config.json` and add the server:

```json
{
  "mcpServers": {
    "preflight": {
      "command": "/path/to/preflight/.venv/bin/preflight",
      "args": ["serve", "--mcp", "--rules", "/path/to/preflight/scenes/kitchen_rules.yaml"]
    }
  }
}
```

Then restart Claude Desktop.

## Example session

These are real tool calls and results from an MCP client, with `--feedback detailed`. The `describe_scene` result is shorter here.

```text
> describe_scene {}
Scene. Positions are in metres in the world frame, and z points up.
Gripper (TCP) at (0.554, 0.000, 1.271), holding nothing.
Workspace: x -0.30 to 0.80, y -0.60 to 0.40, z 0.75 to 1.60.
Objects:
- hot_cup at (0.250, 0.150, 0.800) [hot, liquid]
- knife at (0.500, 0.080, 0.760) [sharp]
...
Person (do not touch):
- human_hand_left at (0.500, 0.320, 0.790) (hand)
...

> pick {"object": "hot_cup"}
done

> move_to {"xyz": [0.5, 0.32, 0.89], "speed": 1.0}
isError=true
blocked: hot_clearance at t = 0.14 s: hot_cup (hot) comes within 22.2 cm of human_hand_left; it started at 22.7 cm, already below the 30 cm limit, and the action brings it closer

> move_to {"xyz": [0.3, -0.1, 1.0], "speed": 1.0}
done, with a change: slowed 3.6x to a peak of 0.20 m/s, because of slow_near_human
```

The third call tries to hold the hot cup 10 cm above the hand of the person. The firewall blocks it, and the robot does not move. The fourth call moves the cup away from the person, but too fast near the person. The firewall runs it 3.6 times slower.

## Results

- `done`: the call passed the check, and the robot moved.
- `done, with a change: ...`: the robot moved, but more slowly than the call asked.
- `blocked: ...` or `blocked by safety policy`: the robot did not move. The result has `isError` set, so the client knows that the action did not run. The LLM reads the reason and can plan a different move.
- `error: ...`: the arguments were not valid, and nothing ran. The result has `isError` set.

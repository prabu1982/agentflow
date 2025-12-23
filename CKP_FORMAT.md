# CKP Format Support

The Mainframe Agent now supports the **CKP (Checkpoint) JSON format**, which provides advanced features for automation workflows including variables, business rules, validations, exceptions, and recovery flows.

## CKP Format Features

### 1. **Variables and Substitution**
- Support for `{{variable_name}}` placeholders in step values
- Variable validation against schema (type, pattern, enum)
- Required and optional variables
- Default values for optional variables

### 2. **Business Rules**
- Rules that apply to specific steps
- Conditional logic based on variable values
- Constraints (hard_block, soft_prompt)
- Default value assignment

### 3. **Validations**
- Post-step validation checks
- Screen text validation
- Exception routing on validation failure

### 4. **Exception Detection**
- Automatic exception detection based on screen text
- Multiple trigger patterns (contains, contains_any)
- Severity levels (recoverable, non-recoverable)

### 5. **Recovery Flows**
- Automatic recovery flow execution for exceptions
- Recovery steps with variable substitution
- Required secrets validation
- Recovery result tracking

## Usage

### Basic CKP File Execution

```bash
python main.py ckp.json --var account_number=123456789 --var account_type=checking
```

### With Variables File

Create a `variables.json` file:
```json
{
  "account_number": "123456789",
  "account_type": "checking",
  "state": "CA",
  "tpx_command": "tpx"
}
```

Then run:
```bash
python main.py ckp.json --vars-file variables.json
```

### Combining File and CLI Variables

CLI variables override file variables:
```bash
python main.py ckp.json --vars-file variables.json --var state=UT
```

## CKP File Structure

```json
{
  "procedure_id": "SETTLEMENT_REQUEST_MAINFRAME",
  "description": "Procedure description",
  "base_steps": [
    {
      "step_id": "S1",
      "action": "launch_application",
      "element": "application",
      "value": "{{application_path}}",
      "application_type": "mainframe"
    }
  ],
  "variables_schema": {
    "required": {
      "account_number": {
        "type": "string",
        "pattern": "^[0-9]{9,18}$"
      }
    },
    "optional": {
      "account_type": {
        "type": "string",
        "enum": ["checking", "savings"],
        "default": "checking"
      }
    }
  },
  "rules": [
    {
      "rule_id": "R-STATE-CHK-01",
      "applies_to_step": "S6",
      "condition": {
        "field": "account_type",
        "operator": "equals",
        "value": "checking"
      },
      "constraint": {
        "field": "state",
        "allowed_values": ["CA", "UT"]
      },
      "severity": "hard_block"
    }
  ],
  "validations": [
    {
      "validation_id": "V-REQ-ACCEPTED-01",
      "step_id": "S7",
      "type": "screen_text_check",
      "expected_text_any": ["Request Accepted", "Settlement Request Created"],
      "failure_routes_to_exception": "REQUEST_NOT_ACCEPTED"
    }
  ],
  "exceptions": [
    {
      "exception_code": "SIGNON_FAILURE",
      "trigger": {
        "type": "screen_text",
        "contains": "Sign-On failure"
      },
      "severity": "recoverable"
    }
  ],
  "recovery_flows": [
    {
      "recovery_id": "REC-SIGNON-01",
      "applies_to_exception": "SIGNON_FAILURE",
      "steps": [
        { "action": "send_keys", "element": "special_key", "special_keys": "CLEAR" },
        { "action": "send_keys", "element": "data_field", "value": "{{user_id}}" },
        { "action": "send_keys", "element": "data_field", "value": "{{password}}" }
      ],
      "required_secrets": ["user_id", "password"]
    }
  ]
}
```

## Workflow Execution Flow

1. **Load CKP File** → Parse procedure, steps, rules, validations, exceptions, recovery flows
2. **Validate Variables** → Check against schema (required, type, pattern, enum)
3. **For Each Step**:
   - **Apply Rules** → Evaluate business rules and apply constraints/effects
   - **Substitute Variables** → Replace `{{variable}}` placeholders
   - **Execute Step** → Call MCP Server tool
   - **Validate Result** → Check validation criteria
   - **Check Exceptions** → Detect exception triggers
   - **Execute Recovery** → Run recovery flow if exception detected
   - **Handle Errors** → Use OLLAMA for error recovery if needed

## Step Format

CKP steps use a different format than simple JSON:

```json
{
  "step_id": "S1",
  "action": "send_keys",
  "element": "data_field",
  "value": "{{account_number}}",
  "application_type": "mainframe",
  "special_keys": "",
  "windowName": "SETTLEMENT REQUEST PANEL",
  "fieldName": "Account Number",
  "screenName": "SETTLEMENT_REQ",
  "startX": "10",
  "startY": "20",
  "endX": "10",
  "endY": "35"
}
```

The agent automatically converts CKP format to MCP tool format.

## Variable Substitution

Variables are substituted in step `value` fields:

- `{{account_number}}` → Replaced with actual account number
- `{{state}}` → Replaced with state value
- Missing required variables → Error
- Missing optional variables → Use default or warn

## Business Rules

Rules are evaluated before step execution:

- **Condition**: Evaluates against variables
- **Constraint**: Enforces allowed values (hard_block stops execution)
- **Effect**: Sets default values for missing variables

## Validations

Validations run after successful step execution:

- **screen_text_check**: Checks if expected text appears in screen
- **failure_routes_to_exception**: Routes to exception handling if validation fails

## Exception Detection

Exceptions are detected by:

- Screen text patterns (contains, contains_any)
- Validation failures that route to exceptions
- Execution errors

## Recovery Flows

Recovery flows execute automatically when exceptions are detected:

- Steps with variable substitution
- Required secrets validation
- Success/failure tracking
- Automatic retry or continuation

## Example: Complete Workflow

```bash
# 1. Create variables file
cat > vars.json << EOF
{
  "account_number": "123456789",
  "account_type": "checking",
  "state": "CA",
  "tpx_command": "tpx",
  "user_id": "user123",
  "password": "pass123"
}
EOF

# 2. Run with CKP file
python main.py ckp.json --vars-file vars.json

# 3. Or use CLI variables
python main.py ckp.json \
  --var account_number=123456789 \
  --var account_type=checking \
  --var state=CA \
  --var user_id=user123 \
  --var password=pass123
```

## Backward Compatibility

The agent still supports simple JSON format files. It automatically detects:
- **CKP format**: Contains `procedure_id` and `base_steps`
- **Simple format**: Array of steps or object with `steps` key

Simple format files work exactly as before, without CKP features.

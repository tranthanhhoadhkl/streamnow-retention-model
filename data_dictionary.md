# StreamNow — Data Dictionary

---

## subscribers.csv
One row per subscriber. This is the primary table.

| Column | Type | Description |
|---|---|---|
| `subscriber_id` | string | Unique subscriber identifier (e.g. `S000001`) |
| `signup_date` | date | Date the subscriber first activated their account |
| `plan` | string | Current plan: `basic`, `standard`, or `premium` |
| `payment_method` | string | `credit_card`, `debit_card`, `paypal`, `voucher` |
| `age` | int | Subscriber age in years |
| `gender` | string | `M`, `F`, `other` |
| `country` | string | `UK` or `IE` |
| `device_primary` | string | Primary device used: `tv`, `mobile`, `tablet`, `laptop` |
| `churned_30d` | int | **Target variable** — 1 if subscriber cancelled between snapshot date and 30 days later, 0 otherwise |

---

## viewing_history.csv
One row per viewing session. A subscriber may have many sessions (or none).

| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique session identifier |
| `subscriber_id` | string | Foreign key → `subscribers.csv` |
| `session_date` | date | Date of the viewing session |
| `duration_mins` | float | Length of the session in minutes |
| `content_type` | string | `movie`, `series`, `documentary`, `kids`, `sport` |
| `completed` | int | 1 if the subscriber finished the content, 0 if they abandoned mid-way |
| `device` | string | Device used for this session |

---

## support_tickets.csv
One row per support interaction. Not all subscribers have raised a ticket.

| Column | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `subscriber_id` | string | Foreign key → `subscribers.csv` |
| `ticket_date` | date | Date ticket was raised |
| `category` | string | `billing`, `technical`, `content`, `cancellation_intent`, `other` |
| `resolved` | int | 1 if the ticket was marked resolved, 0 if still open or unresolved |
| `resolution_days` | float | Days taken to resolve (NaN if unresolved) |

---

## billing.csv
One row per billing cycle per subscriber (up to 12 months of history).

| Column | Type | Description |
|---|---|---|
| `billing_id` | string | Unique billing record identifier |
| `subscriber_id` | string | Foreign key → `subscribers.csv` |
| `billing_date` | date | Date of the billing attempt |
| `amount_due` | float | Amount charged (in £) |
| `payment_status` | string | `paid`, `failed`, `disputed` |
| `retry_count` | int | Number of payment retry attempts (0 if paid first time) |

---

## Notes for Students

- `viewing_history.csv` contains sessions for the **last 90 days** before snapshot date only.
- `support_tickets.csv` contains tickets for the **last 6 months** before snapshot date.
- `billing.csv` contains up to **12 months** of billing history.
- Subscribers with no viewing sessions in the last 90 days are **not missing** — they genuinely had no activity. This is an important signal.
- The `cancellation_intent` category in `support_tickets` is a very strong churn signal — treat it carefully to avoid leakage. Think about whether to include it and justify your decision.

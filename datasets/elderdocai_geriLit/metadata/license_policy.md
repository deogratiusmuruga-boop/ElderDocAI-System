# ElderDocAI-GeriLit — License Policy (dev-v0.1)

## Principle
"PMC Open Access" does NOT equal unrestricted redistribution. Per-article license metadata is
inspected and used for classification. This policy is a technical/documentation policy, not
legal advice.

## License categories
| Category | Definition (from article NLM XML `<license>` / oa.fcgi) |
|---|---|
| PERMISSIVE | Creative Commons CC BY, CC0/public domain, or clearly permissive equivalents (no `nc`, no `nd` modifiers) |
| RESTRICTED | Creative Commons CC BY-NC / CC BY-ND / CC BY-NC-ND or other non-commercial/no-derivatives terms |
| UNKNOWN | License element absent, ambiguous, or unparseable |
| EXCLUDED | Articles that fail integrity checks or carry undisclosed/unsafe redistribution terms |

## Policy for the development corpus
- Accept into the redistributable dev artifact ONLY articles classified **PERMISSIVE**.
- RESTRICTED and UNKNOWN are **excluded** from the accepted dev corpus (recorded with E4/E5).
- The license string and category are recorded per article in the metadata schema
  (`license`, `license_category`).
- If license metadata cannot be verified for an article, it is marked UNKNOWN and not silently
  included in a corpus intended for redistribution.

## Implementation
- `oa.fcgi` license type (Commercial/Non-Commercial) is a pre-check.
- Final category is derived deterministically from the article XML `<license xlink:href=...>`:
  - href/ text contains `creativecommons.org/licenses/by/` and no `/nc-` and no `-nd` → PERMISSIVE;
  - contains `cc0` or "public domain" → PERMISSIVE;
  - contains `nc-` or `-nd` → RESTRICTED;
  - otherwise → UNKNOWN.
- Both the raw license string and category are stored in `metadata/articles_metadata.json` and the manifests.

## Final-study note
The final corpus remains governed by per-article license terms; this policy will be reapplied at
final-corpus build time with the same conservative (PERMISSIVE-only) inclusion rule unless the
research team explicitly authorizes a licensed-derived-artifact exception.
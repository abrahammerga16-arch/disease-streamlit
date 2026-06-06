import requests as _requests

def _supabase_signup(role: str, user_id: str, name: str = "") -> tuple[bool, str]:
    uid = user_id.strip().upper()
    if not uid:
        return False, "User ID cannot be empty."
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

        # Check if user already exists
        check = _requests.get(
            f"{SUPABASE_URL}/rest/v1/app_users?user_id=eq.{uid}&select=user_id",
            headers=headers, timeout=8,
        )
        if check.ok and check.json():
            return False, "⚠️ User ID already exists. Please log in instead."

        # Insert new user
        insert = _requests.post(
            f"{SUPABASE_URL}/rest/v1/app_users",
            headers=headers,
            json={"user_id": uid, "role": role, "name": name.strip()},
            timeout=8,
        )
        if insert.status_code in (200, 201):
            return True, f"✅ Account created! Your ID: **{uid}**"
        return False, f"❌ Insert failed: {insert.text}"

    except Exception as e:
        return False, f"❌ Signup error: {e}"


def _supabase_login(role: str, user_id: str) -> tuple[bool, str]:
    uid = user_id.strip().upper()
    if not uid:
        return False, "User ID cannot be empty."
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        result = _requests.get(
            f"{SUPABASE_URL}/rest/v1/app_users?user_id=eq.{uid}&role=eq.{role}&select=user_id,name",
            headers=headers, timeout=8,
        )
        if result.ok and result.json():
            name = result.json()[0].get("name", "")
            greeting = f" ({name})" if name else ""
            return True, f"✅ Welcome back{greeting}!"
        return False, "❌ Invalid ID or role mismatch."

    except Exception as e:
        return False, f"❌ Login error: {e}"

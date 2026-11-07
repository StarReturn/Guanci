"""SMS service — supports mock (dev) + Aliyun SMS (prod).

For Tencent Cloud SMS or others, add a new branch in ``send``.
"""
from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class SmsService:
    def send(self, *, phone: str, code: str) -> bool:
        """Send a verification code via configured provider."""
        provider = (settings.sms_provider or "mock").lower()
        if provider == "mock":
            return self._send_mock(phone, code)
        if provider == "aliyun":
            return self._send_aliyun(phone, code)
        if provider == "tencent":
            # TODO: implement Tencent SMS
            logger.warning("sms_provider=tencent not implemented, falling back to mock")
            return self._send_mock(phone, code)
        logger.warning("unknown sms_provider=%r, falling back to mock", provider)
        return self._send_mock(phone, code)

    def _send_mock(self, phone: str, code: str) -> bool:
        """Dev mode: log the code instead of sending it.

        In non-prod env, the SendCodeResponse will include ``dev_code``
        so Postman / curl users can fetch the code without a real SMS.
        """
        logger.info("SMS_MOCK phone=%s code=%s", phone, code)
        return True

    def _send_aliyun(self, phone: str, code: str) -> bool:
        """Send via Aliyun DySMS API.

        Requires ``sms_aliyun_*`` env vars. Uses bare HTTP request
        rather than the aliyun-python-sdk-core package because we
        don't want the extra dependency; the API is one signed POST.
        """
        try:
            import hmac
            import hashlib
            import base64
            import urllib.parse
            import urllib.request
            import uuid
            from datetime import datetime, timezone

            params = {
                "PhoneNumbers": phone,
                "SignName": settings.sms_aliyun_sign_name,
                "TemplateCode": settings.sms_aliyun_template_code,
                "TemplateParam": f'{{"code":"{code}"}}',
                "Action": "SendSms",
                "Version": "2017-05-25",
                "Format": "JSON",
                "RegionId": "cn-hangzhou",
                "AccessKeyId": settings.sms_aliyun_access_key,
                "SignatureMethod": "HMAC-SHA1",
                "SignatureNonce": str(uuid.uuid4()),
                "SignatureVersion": "1.0",
                "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            # CanonicalizedQueryString (sorted, url-encoded)
            sorted_items = sorted(params.items())
            canonical = "&".join(
                f"{urllib.parse.quote(k, safe='~')}={urllib.parse.quote(v, safe='~')}"
                for k, v in sorted_items
            )
            string_to_sign = "GET&" + urllib.parse.quote("/", safe="") + "&" + urllib.parse.quote(canonical, safe="")
            signature = base64.b64encode(
                hmac.new(
                    (settings.sms_aliyun_access_secret + "&").encode(),
                    string_to_sign.encode(),
                    hashlib.sha1,
                ).digest()
            ).decode()

            params["Signature"] = signature
            url = "https://dysmsapi.aliyuncs.com/?" + urllib.parse.urlencode(params)

            with urllib.request.urlopen(url, timeout=10) as resp:
                import json
                body = json.loads(resp.read())
                if body.get("Code") != "OK":
                    logger.error("aliyun_sms_failed phone=%s body=%s", phone, body)
                    return False
                return True
        except Exception as exc:
            logger.exception("aliyun_sms_exception phone=%s", phone)
            return False


sms_service = SmsService()


__all__ = ["sms_service"]

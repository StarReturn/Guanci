"""数字人分享（第二批 A）：生成/撤销分享码、兑换、被分享列表。

规则（甲方已定 2026-10-12）：
- 默认可分享，无数字人类型限制
- 分享码 6 位（0-9/A-Z 去混淆字符），二维码内容 = 分享码
- 被分享者：仅对话权（不可改人设/换背景/重建/删除）
- 免费
- 配额谁用谁付：实时对话 entitlement 按 state.user_id 计量（现状即满足）
- 对话记录/长期记忆按用户隔离（dialogue_sessions / avatar_memories 均含
  user_id，现状即满足）
"""
from __future__ import annotations

import secrets
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.database import SessionLocal
from sqlalchemy import text

router = APIRouter()

# 去混淆字符集（无 0/O/1/I/L）
_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


def _uid(request: Request) -> str:
    from app.middleware.jwt import get_request_user_id
    uid = get_request_user_id(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="user login required")
    return uid


def _gen_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


# ---------- DTO ----------

class ShareCreateRead(BaseModel):
    share_id: str
    share_code: str
    digital_human_id: str
    created_at: str


class SharedWithMeRead(BaseModel):
    digital_human_id: str
    display_name: str
    owner_nickname: str
    has_photo: bool = False
    is_ready: bool


class RedeemRead(BaseModel):
    digital_human_id: str
    display_name: str
    owner_nickname: str


# ---------- 端点 ----------

@router.post("/{digital_human_id}/share", response_model=ShareCreateRead, status_code=201)
async def create_share(digital_human_id: str, request: Request) -> ShareCreateRead:
    """创建者生成分享码（同一数字人同刻最多 5 个未兑换码，防刷）。"""
    uid = _uid(request)
    with SessionLocal() as db:
        dh = db.execute(text(
            "select display_name from digital_humans "
            "where digital_human_id=:d and owner_user_id=:u and deleted_at is null"
        ), {"d": digital_human_id, "u": uid}).fetchone()
        if dh is None:
            raise HTTPException(status_code=404, detail="数字人不存在")
        active = db.execute(text(
            "select count(*) from dh_shares "
            "where digital_human_id=:d and owner_user_id=:u and status='active' "
            "and grantee_user_id is null"
        ), {"d": digital_human_id, "u": uid}).fetchone()[0]
        if active >= 5:
            raise HTTPException(status_code=409, detail="该数字人待使用的分享码已达上限（5 个）")
        share_id = str(uuid4())
        code = _gen_code()
        # 码全局唯一，冲突重试
        for _ in range(5):
            dup = db.execute(text(
                "select 1 from dh_shares where share_code=:c and status='active'"
            ), {"c": code}).fetchone()
            if dup is None:
                break
            code = _gen_code()
        db.execute(text(
            "insert into dh_shares (share_id, digital_human_id, owner_user_id, share_code) "
            "values (:s, :d, :u, :c)"
        ), {"s": share_id, "d": digital_human_id, "u": uid, "c": code})
        db.commit()
    return ShareCreateRead(
        share_id=share_id, share_code=code,
        digital_human_id=digital_human_id, created_at="",
    )


@router.get("/shared-with-me", response_model=list[SharedWithMeRead])
async def list_shared_with_me(request: Request) -> list[SharedWithMeRead]:
    """我添加的（别人分享给我的）数字人列表。"""
    uid = _uid(request)
    with SessionLocal() as db:
        rows = db.execute(text("""
            select s.digital_human_id, d.display_name,
                   coalesce(o.nickname, '未知用户') as owner_nick,
                   d.photo_storage_key is not null as has_photo,
                   (d.creation_status = 'ready') as is_ready
            from dh_shares s
            join digital_humans d on d.digital_human_id = s.digital_human_id
            left join users o on o.user_id = s.owner_user_id
            where s.grantee_user_id = :u and s.status = 'redeemed'
              and d.deleted_at is null
            order by s.redeemed_at desc
        """), {"u": uid}).fetchall()
        return [
            SharedWithMeRead(
                digital_human_id=r[0], display_name=r[1], owner_nickname=r[2],
                has_photo=bool(r[3]), is_ready=bool(r[4]),
            ) for r in rows
        ]


@router.post("/redeem", response_model=RedeemRead, status_code=201)
async def redeem_share(body: RedeemRequest, request: Request) -> RedeemRead:
    """兑换分享码：添加别人分享的数字人到我的列表。"""
    uid = _uid(request)
    code = body.share_code.strip().upper()
    with SessionLocal() as db:
        share = db.execute(text(
            "select share_id, digital_human_id, owner_user_id, status, grantee_user_id "
            "from dh_shares where share_code=:c"
        ), {"c": code}).fetchone()
        if share is None:
            raise HTTPException(status_code=404, detail="分享码不存在，请核对后重试")
        if share[3] == "revoked":
            raise HTTPException(status_code=410, detail="该分享码已被创建者撤销")
        if share[3] == "redeemed":
            if share[4] == uid:
                raise HTTPException(status_code=409, detail="你已经添加过这个数字人了")
            raise HTTPException(status_code=410, detail="该分享码已被使用")
        if share[2] == uid:
            raise HTTPException(status_code=422, detail="不能添加自己分享的数字人")
        dh = db.execute(text(
            "select display_name from digital_humans "
            "where digital_human_id=:d and deleted_at is null"
        ), {"d": share[1]}).fetchone()
        if dh is None:
            raise HTTPException(status_code=410, detail="该数字人已被删除")
        owner_nick = db.execute(text(
            "select coalesce(nickname,'未知用户') from users where user_id=:o"
        ), {"o": share[2]}).fetchone()[0]
        db.execute(text(
            "update dh_shares set status='redeemed', grantee_user_id=:u, redeemed_at=now() "
            "where share_id=:s"
        ), {"u": uid, "s": share[0]})
        db.commit()
        return RedeemRead(digital_human_id=share[1], display_name=dh[0], owner_nickname=owner_nick)


class RedeemRequest(BaseModel):
    share_code: str = Field(min_length=4, max_length=12)


@router.delete("/{digital_human_id}/share", status_code=204)
async def revoke_my_shares(digital_human_id: str, request: Request) -> None:
    """创建者撤销该数字人的全部分享码（已兑换的解除共享关系）。"""
    uid = _uid(request)
    with SessionLocal() as db:
        own = db.execute(text(
            "select 1 from digital_humans where digital_human_id=:d and owner_user_id=:u"
        ), {"d": digital_human_id, "u": uid}).fetchone()
        if own is None:
            raise HTTPException(status_code=404, detail="数字人不存在")
        db.execute(text(
            "update dh_shares set status='revoked', revoked_at=now() "
            "where digital_human_id=:d and owner_user_id=:u and status != 'revoked'"
        ), {"d": digital_human_id, "u": uid})
        db.commit()


@router.delete("/shared-with-me/{digital_human_id}", status_code=204)
async def remove_shared(digital_human_id: str, request: Request) -> None:
    """被分享者移除（不撤销他人码，仅自己列表移除）。"""
    uid = _uid(request)
    with SessionLocal() as db:
        db.execute(text(
            "update dh_shares set status='revoked', revoked_at=now() "
            "where grantee_user_id=:u and digital_human_id=:d and status='redeemed'"
        ), {"u": uid, "d": digital_human_id})
        db.commit()

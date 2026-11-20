"""Small synthetic dataset for tests, demos and smoke runs.

These are NOT real Bitcoin Core commits. They are hand-written bug/fix pairs
modelled on common Bitcoin Core fault classes. Commit SHAs are fake
(`5e5e...` prefix) and every record has `source="synthetic"`.
"""

from __future__ import annotations

import hashlib

from .mining import function_diff
from .records import CONFIRMED, BugRecord, make_record_id

_BUGS = [
    ("src/txmempool.cpp", "CTxMemPool::TrimToSize", "mempool: fix off-by-one when trimming to size",
     """void CTxMemPool::TrimToSize(size_t sizelimit)
{
    while (!mapTx.empty() && DynamicMemoryUsage() >= sizelimit) {
        RemoveWorst();
    }
}""",
     """void CTxMemPool::TrimToSize(size_t sizelimit)
{
    while (!mapTx.empty() && DynamicMemoryUsage() > sizelimit) {
        RemoveWorst();
    }
}"""),
    ("src/net_processing.cpp", "PeerManagerImpl::ProcessGetData", "net: take cs_main before reading the block index",
     """void PeerManagerImpl::ProcessGetData(CNode& pfrom)
{
    const CBlockIndex* pindex = m_chainman.m_blockman.LookupBlockIndex(hash);
    if (pindex && pindex->nStatus & BLOCK_HAVE_DATA) {
        SendBlock(pfrom, *pindex);
    }
}""",
     """void PeerManagerImpl::ProcessGetData(CNode& pfrom)
{
    LOCK(cs_main);
    const CBlockIndex* pindex = m_chainman.m_blockman.LookupBlockIndex(hash);
    if (pindex && pindex->nStatus & BLOCK_HAVE_DATA) {
        SendBlock(pfrom, *pindex);
    }
}"""),
    ("src/wallet/spend.cpp", "SelectCoinsBnB", "wallet: fix integer overflow in selection target",
     """bool SelectCoinsBnB(std::vector<OutputGroup>& utxo_pool, const CAmount& selection_target)
{
    int curr_value = 0;
    for (const OutputGroup& group : utxo_pool) {
        curr_value += group.GetSelectionAmount();
    }
    return curr_value >= selection_target;
}""",
     """bool SelectCoinsBnB(std::vector<OutputGroup>& utxo_pool, const CAmount& selection_target)
{
    CAmount curr_value = 0;
    for (const OutputGroup& group : utxo_pool) {
        curr_value += group.GetSelectionAmount();
    }
    return curr_value >= selection_target;
}"""),
    ("src/script/interpreter.cpp", "CheckSignatureEncoding", "script: reject empty signature before reading hashtype",
     """bool CheckSignatureEncoding(const std::vector<unsigned char>& vchSig, unsigned int flags, ScriptError* serror)
{
    if ((flags & SCRIPT_VERIFY_STRICTENC) != 0 && !IsDefinedHashtypeSignature(vchSig)) {
        return set_error(serror, SCRIPT_ERR_SIG_HASHTYPE);
    }
    return true;
}""",
     """bool CheckSignatureEncoding(const std::vector<unsigned char>& vchSig, unsigned int flags, ScriptError* serror)
{
    if (vchSig.size() == 0) {
        return true;
    }
    if ((flags & SCRIPT_VERIFY_STRICTENC) != 0 && !IsDefinedHashtypeSignature(vchSig)) {
        return set_error(serror, SCRIPT_ERR_SIG_HASHTYPE);
    }
    return true;
}"""),
    ("src/validation.cpp", "CheckSequenceLocksAtTip", "validation: use the next block height for sequence locks",
     """bool CheckSequenceLocksAtTip(CBlockIndex* tip, const LockPoints& lock_points)
{
    int height = tip->nHeight;
    return lock_points.height < height;
}""",
     """bool CheckSequenceLocksAtTip(CBlockIndex* tip, const LockPoints& lock_points)
{
    int height = tip->nHeight + 1;
    return lock_points.height < height;
}"""),
    ("src/addrman.cpp", "AddrManImpl::Select_", "addrman: fix out-of-bounds bucket position",
     """std::pair<CAddress, NodeSeconds> AddrManImpl::Select_(bool new_only) const
{
    int nKBucketPos = insecure_rand.randbits(ADDRMAN_BUCKET_SIZE_LOG2 + 1);
    int nId = GetEntry(new_only, nKBucket, nKBucketPos);
    return {mapInfo.at(nId), mapInfo.at(nId).nLastTry};
}""",
     """std::pair<CAddress, NodeSeconds> AddrManImpl::Select_(bool new_only) const
{
    int nKBucketPos = insecure_rand.randbits(ADDRMAN_BUCKET_SIZE_LOG2);
    int nId = GetEntry(new_only, nKBucket, nKBucketPos);
    return {mapInfo.at(nId), mapInfo.at(nId).nLastTry};
}"""),
    ("src/policy/fees.cpp", "TxConfirmStats::Record", "fees: guard against negative bucket index",
     """void TxConfirmStats::Record(int blocksToConfirm, double feerate)
{
    unsigned int bucketindex = bucketMap.lower_bound(feerate)->second;
    for (size_t i = blocksToConfirm; i <= confAvg.size(); i++) {
        confAvg[i - 1][bucketindex]++;
    }
}""",
     """void TxConfirmStats::Record(int blocksToConfirm, double feerate)
{
    if (blocksToConfirm < 1) return;
    unsigned int bucketindex = bucketMap.lower_bound(feerate)->second;
    for (size_t i = blocksToConfirm; i <= confAvg.size(); i++) {
        confAvg[i - 1][bucketindex]++;
    }
}"""),
    ("src/rpc/blockchain.cpp", "getblockhash", "rpc: fix wrong comparison in height range check",
     """static RPCHelpMan getblockhash()
{
    int nHeight = request.params[0].getInt<int>();
    if (nHeight < 0 || nHeight >= active_chain.Height()) {
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Block height out of range");
    }
    return active_chain[nHeight]->GetBlockHash().GetHex();
}""",
     """static RPCHelpMan getblockhash()
{
    int nHeight = request.params[0].getInt<int>();
    if (nHeight < 0 || nHeight > active_chain.Height()) {
        throw JSONRPCError(RPC_INVALID_PARAMETER, "Block height out of range");
    }
    return active_chain[nHeight]->GetBlockHash().GetHex();
}"""),
]


def synthetic_records() -> list[BugRecord]:
    from .cpp import subsystem_for_path

    out = []
    for path, func, subject, buggy, fixed in _BUGS:
        sha = "5e5e" + hashlib.sha1(f"{path}:{func}".encode()).hexdigest()[4:]
        out.append(BugRecord(
            id=make_record_id(sha, path, func), commit_sha=sha, parent_sha="",
            file_path=path, function_name=func, commit_message=f"{subject}\n\nSYNTHETIC EXAMPLE - not a real commit.",
            buggy_code=buggy, fixed_code=fixed, diff=function_diff(buggy, fixed, path),
            subsystem=subsystem_for_path(path), label=CONFIRMED, reviewer="synthetic",
            source="synthetic", tags=["synthetic"],
        ))
    return out

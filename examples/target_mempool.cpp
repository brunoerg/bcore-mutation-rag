// Synthetic target for demos: shaped like Bitcoin Core code, not a copy of it.
#include <cstddef>

void CTxMemPool::LimitSize(size_t limit, size_t& removed)
{
    removed = 0;
    while (!mapTx.empty() && DynamicMemoryUsage() > limit) {
        RemoveWorst();
        ++removed;
    }
    if (removed > 0 && m_opts.signals) {
        m_opts.signals->MempoolTrimmed(removed);
    }
}

<script setup>
// 课栈 - 编程学习助手（阶段 4 + 6.x + UI 优化）
// 现代 AI 聊天风格：Markdown 渲染、思考折叠、打字光标、头像、动效
import { ref, computed, onMounted } from 'vue'
import { marked } from 'marked'

const messages = ref([])        // {role, content, thinking, tools, collapsed}
const input = ref('')
const loading = ref(false)
const chatBox = ref(null)
const threads = ref([])
const activeThread = ref('')
const editingId = ref(null)
const editingTitle = ref('')
const THREAD_KEY = 'kezhan_thread_id'

// Markdown 渲染（AI 回复是 markdown）
marked.setOptions({ breaks: true, gfm: true })
const renderMd = (text) => marked.parse(text || '')

// 消息附带 render 后的 html（v-html 用），避免每次渲染都 parse
function attachRender(m) {
  if (m.role === 'assistant' && m.content && !m._html) {
    m._html = renderMd(m.content)
  }
  return m
}

function scrollBottom() {
  setTimeout(() => {
    if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
  }, 50)
}

// 思考区折叠（默认展开最新一条，历史折叠）
function toggleThinking(m) {
  m.collapsed = !m.collapsed
}

async function loadThreads() {
  try {
    const resp = await fetch('/threads')
    if (resp.ok) threads.value = await resp.json()
  } catch (e) { /* ignore */ }
}

async function switchThread(tid) {
  if (loading.value) return
  activeThread.value = tid
  localStorage.setItem(THREAD_KEY, tid)
  messages.value = []
  try {
    const resp = await fetch(`/threads/${encodeURIComponent(tid)}/messages`)
    if (resp.ok) {
      const data = await resp.json()
      messages.value = data.messages.map((m) => {
        const item = {
          role: m.role === 'human' ? 'user' : 'assistant',
          content: m.content,
          thinking: m.thinking || '',
          tools: [],
          collapsed: !m.thinking,  // 有思考的历史消息默认折叠
        }
        return attachRender(item)
      })
      scrollBottom()
    }
  } catch (e) { /* ignore */ }
}

async function newSession() {
  if (loading.value) return
  activeThread.value = crypto.randomUUID()
  localStorage.setItem(THREAD_KEY, activeThread.value)
  messages.value = []
  await loadThreads()
}

function startRename(t) {
  editingId.value = t.thread_id
  editingTitle.value = t.title
}

async function saveRename(tid) {
  const title = editingTitle.value.trim()
  editingId.value = null
  if (!title) return
  try {
    await fetch(`/threads/${encodeURIComponent(tid)}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
    await loadThreads()
  } catch (e) { /* ignore */ }
}

async function removeThread(tid) {
  if (loading.value) return
  if (!confirm('确定删除该会话吗？删除后不可恢复。')) return
  try {
    await fetch(`/threads/${encodeURIComponent(tid)}`, { method: 'DELETE' })
    if (activeThread.value === tid) await newSession()
    else await loadThreads()
  } catch (e) { /* ignore */ }
}

// 示例问题快速填充
function useExample(q) {
  input.value = q
}

// 导出当前会话为 Markdown（阶段 7 需求）
async function exportChat() {
  if (!activeThread.value || loading.value) return
  try {
    const resp = await fetch(`/threads/${encodeURIComponent(activeThread.value)}/export`)
    if (!resp.ok) throw new Error('导出失败')
    const data = await resp.json()
    const blob = new Blob([data.markdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${data.title || '会话'}.md`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    alert('导出失败：' + e.message)
  }
}

// 生成学习计划（结构化输出 → 卡片渲染）
async function generatePlan(q) {
  if (loading.value) return
  input.value = ''
  loading.value = true
  messages.value.push({ role: 'user', content: q })
  messages.value.push({ role: 'plan', plan: null, error: '' })
  const idx = messages.value.length - 1
  scrollBottom()
  try {
    const resp = await fetch('/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: q }),
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    messages.value[idx].plan = await resp.json()
  } catch (e) {
    messages.value[idx].error = e.message
  } finally {
    loading.value = false
    scrollBottom()
  }
}

async function send() {
  const text = input.value.trim()
  if (!text || loading.value) return
  input.value = ''
  loading.value = true

  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '', thinking: '', tools: [], collapsed: false })
  // ★ Vue 3 响应式陷阱：从代理数组取引用
  const msg = messages.value[messages.value.length - 1]
  scrollBottom()

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, thread_id: activeThread.value }),
    })
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const chunk = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const line = chunk.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        const evt = JSON.parse(line.slice(5))
        if (evt.type === 'reasoning') msg.thinking += evt.content
        else if (evt.type === 'token') {
          msg.content += evt.content
          attachRender(msg)
        }
        else if (evt.type === 'tool') msg.tools.push(evt)
        else if (evt.type === 'error') msg.content += `\n[错误] ${evt.message}`
        scrollBottom()
      }
    }
  } catch (e) {
    msg.content += `\n[请求失败] ${e.message}`
  } finally {
    loading.value = false
    scrollBottom()
    await loadThreads()
  }
}

onMounted(async () => {
  await loadThreads()
  const saved = localStorage.getItem(THREAD_KEY)
  if (saved && threads.value.some((t) => t.thread_id === saved)) {
    await switchThread(saved)
  } else {
    activeThread.value = crypto.randomUUID()
    localStorage.setItem(THREAD_KEY, activeThread.value)
  }
})
</script>

<template>
  <div class="layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-brand">🎓 课栈</div>
      <button class="new-session" @click="newSession" :disabled="loading">＋ 新建会话</button>
      <div class="thread-list">
        <div
          v-for="t in threads"
          :key="t.thread_id"
          class="thread-item"
          :class="{ active: t.thread_id === activeThread }"
          @click="switchThread(t.thread_id)"
          :title="t.thread_id"
        >
          <input
            v-if="editingId === t.thread_id"
            v-model="editingTitle"
            class="rename-input"
            @click.stop
            @keyup.enter="saveRename(t.thread_id)"
            @keyup.esc="editingId = null"
            @blur="saveRename(t.thread_id)"
          />
          <template v-else>
            <span class="thread-title">{{ t.title }}</span>
            <span class="thread-actions" @click.stop>
              <button class="icon-btn" title="重命名" @click="startRename(t)">✏️</button>
              <button class="icon-btn" title="删除会话" @click="removeThread(t.thread_id)">🗑</button>
            </span>
          </template>
        </div>
        <div v-if="threads.length === 0" class="thread-empty">暂无历史会话</div>
      </div>
    </aside>

    <!-- 聊天区 -->
    <div class="app">
      <header>
        <h1>🎓 课栈 · 编程学习助手</h1>
        <span class="hint">会话记忆 · 思考过程实时展示</span>
        <button class="export-btn" :disabled="loading || !activeThread" @click="exportChat">📥 导出对话</button>
      </header>

      <main ref="chatBox" class="chat">
        <!-- 空状态：示例问题 -->
        <div v-if="messages.length === 0" class="empty">
          <div class="empty-icon">🎓</div>
          <div class="empty-title">你好！我是课栈，编程学习助手</div>
          <div class="empty-sub">可以问我课程推荐、学习路线规划、知识库问答</div>
          <div class="examples">
            <button v-for="q in ['推荐一门 Python 入门课', '有没有 Vue 进阶课程？']"
              :key="q" class="example-btn" @click="useExample(q)">{{ q }}</button>
            <button class="example-btn plan-btn" @click="generatePlan('零基础学 Python，目标数据分析') ">
              📊 生成学习计划（结构化卡片）
            </button>
          </div>
        </div>

        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div v-if="m.role === 'user'" class="user-bubble">{{ m.content }}</div>

          <template v-else>
            <div class="ai-avatar">🤖</div>
            <div class="ai-body">
              <!-- 学习计划卡片（结构化输出） -->
              <div v-if="m.role === 'plan'" class="plan-card">
                <div v-if="!m.plan && !m.error" class="typing">
                  正在规划学习路线<span class="cursor">▍</span>
                </div>
                <div v-else-if="m.error" class="plan-error">计划生成失败：{{ m.error }}</div>
                <template v-else>
                  <div class="plan-header">
                    <span class="plan-goal">🎯 {{ m.plan.goal }}</span>
                    <span class="plan-badge">{{ m.plan.level }}</span>
                  </div>
                  <div class="plan-meta">总时长约 {{ m.plan.total_hours }} 小时 · {{ m.plan.topics.length }} 个主题</div>
                  <ol class="plan-topics">
                    <li v-for="t in m.plan.topics" :key="t.order" class="plan-topic">
                      <span class="plan-order">{{ t.order }}</span>
                      <span class="plan-name">{{ t.name }}</span>
                      <span class="plan-minutes">{{ Math.round(t.minutes / 60 * 10) / 10 }}h</span>
                    </li>
                  </ol>
                </template>
              </div>
              <!-- 思考区：可折叠 -->
              <div v-if="m.thinking" class="thinking" @click="toggleThinking(m)">
                <span class="thinking-label">🤔 思考过程</span>
                <span class="thinking-toggle">{{ m.collapsed ? '展开 ▾' : '收起 ▴' }}</span>
                <div v-if="!m.collapsed" class="thinking-content">{{ m.thinking }}</div>
              </div>
              <!-- 工具调用（仅当前轮实时显示） -->
              <div v-for="(t, j) in m.tools" :key="j" class="tool-call">
                🔧 调用工具 <b>{{ t.name }}</b>
              </div>
              <!-- 回复：Markdown 渲染 -->
              <div v-if="m.content" class="answer markdown" v-html="m._html || renderMd(m.content)"></div>
              <div v-else-if="loading && i === messages.length - 1" class="typing">
                思考中<span class="cursor">▍</span>
              </div>
            </div>
          </template>
        </div>
      </main>

      <footer>
        <input
          v-model="input"
          placeholder="输入你的问题…（Enter 发送）"
          :disabled="loading"
          @keyup.enter="send"
        />
        <button class="send-btn" :disabled="loading || !input.trim()" @click="send">
          {{ loading ? '生成中' : '发送' }}
        </button>
      </footer>
    </div>
  </div>
</template>

<style>
/* ========== 基础 ========== */
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f7f8fa; color: #1f2937; }
.layout { display: flex; height: 100vh; }

/* ========== 侧边栏 ========== */
.sidebar { width: 250px; background: #171923; color: #e2e8f0; display: flex; flex-direction: column; flex-shrink: 0; }
.sidebar-brand { padding: 18px 16px 8px; font-size: 17px; font-weight: 700; letter-spacing: .5px; }
.new-session { margin: 10px 14px 14px; padding: 10px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; border: none; border-radius: 10px; font-size: 14px; cursor: pointer; transition: opacity .2s; }
.new-session:hover { opacity: .9; }
.new-session:disabled { opacity: .5; cursor: not-allowed; }
.thread-list { flex: 1; overflow-y: auto; padding: 0 10px 14px; }
.thread-item { padding: 10px 12px; border-radius: 10px; cursor: pointer; margin-bottom: 3px; display: flex; justify-content: space-between; align-items: center; gap: 6px; transition: background .15s; }
.thread-item:hover { background: #262a3a; }
.thread-item.active { background: #3730a3; }
.thread-title { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.thread-actions { display: none; gap: 2px; flex-shrink: 0; }
.thread-item:hover .thread-actions { display: flex; }
.icon-btn { background: none; border: none; cursor: pointer; font-size: 12px; padding: 2px 4px; border-radius: 4px; opacity: .75; }
.icon-btn:hover { background: rgba(255,255,255,.15); opacity: 1; }
.rename-input { width: 100%; padding: 5px 8px; border: 1px solid #818cf8; border-radius: 6px; font-size: 13px; background: #fff; color: #1e293b; outline: none; }
.thread-empty { color: #64748b; font-size: 13px; text-align: center; padding: 24px 0; }

/* ========== 聊天区 ========== */
.app { flex: 1; display: flex; flex-direction: column; min-width: 0; }
header { padding: 14px 24px; background: rgba(255,255,255,.85); backdrop-filter: blur(8px); border-bottom: 1px solid #eef0f3; display: flex; align-items: baseline; gap: 12px; position: sticky; top: 0; z-index: 5; }
header h1 { font-size: 18px; color: #111827; }
.hint { font-size: 12px; color: #9ca3af; }

.chat { flex: 1; overflow-y: auto; padding: 28px 24px 40px; scroll-behavior: smooth; }

/* 空状态 */
.empty { text-align: center; margin-top: 12vh; animation: fadeUp .5s ease; }
.empty-icon { font-size: 56px; margin-bottom: 14px; }
.empty-title { font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 8px; }
.empty-sub { color: #9ca3af; font-size: 14px; margin-bottom: 28px; }
.examples { display: flex; flex-direction: column; gap: 10px; align-items: center; }
.example-btn { padding: 10px 22px; background: #fff; border: 1px solid #e5e7eb; border-radius: 999px; font-size: 14px; color: #4f46e5; cursor: pointer; transition: all .2s; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
.example-btn:hover { border-color: #6366f1; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99,102,241,.15); }

/* 消息 */
.msg { display: flex; margin-bottom: 22px; animation: fadeUp .3s ease; }
.msg.user { justify-content: flex-end; }
.user-bubble { max-width: 70%; padding: 11px 18px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; border-radius: 18px 18px 4px 18px; font-size: 15px; line-height: 1.65; box-shadow: 0 2px 8px rgba(99,102,241,.25); white-space: pre-wrap; word-break: break-word; }
.ai-avatar { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; margin-right: 12px; box-shadow: 0 2px 6px rgba(99,102,241,.3); }
.ai-body { flex: 1; min-width: 0; padding-top: 4px; }

/* 思考区（可折叠） */
.thinking { background: #f8fafc; border: 1px solid #eef1f5; border-radius: 12px; padding: 10px 14px; margin-bottom: 10px; cursor: pointer; transition: background .15s; }
.thinking:hover { background: #f1f5f9; }
.thinking-label { font-size: 12px; color: #64748b; font-weight: 600; }
.thinking-toggle { float: right; font-size: 11px; color: #94a3b8; }
.thinking-content { margin-top: 8px; font-size: 13px; color: #64748b; line-height: 1.7; max-height: 160px; overflow-y: auto; white-space: pre-wrap; }

/* 工具调用 */
.tool-call { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #0369a1; background: #f0f9ff; border-radius: 8px; padding: 6px 12px; margin-bottom: 8px; }

/* 回复（Markdown） */
.answer { font-size: 15px; line-height: 1.75; word-break: break-word; }
.markdown :deep(p) { margin: 0 0 10px; }
.markdown :deep(h1), .markdown :deep(h2), .markdown :deep(h3) { margin: 16px 0 8px; font-weight: 600; }
.markdown :deep(ul), .markdown :deep(ol) { margin: 0 0 10px; padding-left: 22px; }
.markdown :deep(li) { margin: 4px 0; }
.markdown :deep(code) { background: #f1f5f9; padding: 2px 6px; border-radius: 5px; font-size: 13px; font-family: Consolas, monospace; }
.markdown :deep(pre) { background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 10px; overflow-x: auto; margin: 0 0 12px; }
.markdown :deep(pre code) { background: none; padding: 0; color: inherit; }
.markdown :deep(strong) { font-weight: 600; }
.markdown :deep(a) { color: #4f46e5; }
.markdown :deep(table) { border-collapse: collapse; margin: 10px 0; }
.markdown :deep(th), .markdown :deep(td) { border: 1px solid #e5e7eb; padding: 6px 12px; font-size: 13px; }
.markdown :deep(blockquote) { border-left: 3px solid #6366f1; padding-left: 12px; color: #6b7280; margin: 10px 0; }

/* 打字状态 */
.typing { color: #9ca3af; font-size: 14px; }
.cursor { display: inline-block; animation: blink 1s step-end infinite; color: #6366f1; }

/* 导出按钮 */
.export-btn { margin-left: auto; padding: 7px 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 13px; color: #4f46e5; cursor: pointer; transition: all .2s; }
.export-btn:hover:not(:disabled) { border-color: #6366f1; background: #f5f3ff; }
.export-btn:disabled { opacity: .5; cursor: not-allowed; }

/* 学习计划卡片 */
.plan-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px 20px; box-shadow: 0 2px 10px rgba(0,0,0,.05); margin-bottom: 10px; }
.plan-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.plan-goal { font-size: 16px; font-weight: 600; color: #1f2937; }
.plan-badge { padding: 2px 10px; border-radius: 999px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; font-size: 12px; flex-shrink: 0; }
.plan-meta { font-size: 12px; color: #9ca3af; margin-bottom: 12px; }
.plan-topics { list-style: none; padding: 0; margin: 0; }
.plan-topic { display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px dashed #f0f0f3; }
.plan-topic:last-child { border-bottom: none; }
.plan-order { width: 22px; height: 22px; border-radius: 50%; background: #eef2ff; color: #4f46e5; font-size: 12px; font-weight: 600; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.plan-name { flex: 1; font-size: 14px; color: #334155; }
.plan-minutes { font-size: 12px; color: #94a3b8; flex-shrink: 0; }
.plan-error { color: #dc2626; font-size: 14px; }
.example-btn.plan-btn { border-style: dashed; border-color: #c7d2fe; background: #f5f3ff; }

/* 底部输入 */
footer { display: flex; gap: 10px; padding: 16px 24px 20px; background: linear-gradient(transparent, #f7f8fa 30%); }
footer input { flex: 1; padding: 13px 18px; border: 1px solid #e5e7eb; border-radius: 12px; font-size: 15px; outline: none; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,.05); transition: border-color .2s, box-shadow .2s; }
footer input:focus { border-color: #6366f1; box-shadow: 0 2px 12px rgba(99,102,241,.15); }
.send-btn { padding: 13px 28px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; border: none; border-radius: 12px; font-size: 15px; cursor: pointer; transition: opacity .2s, transform .1s; }
.send-btn:hover:not(:disabled) { opacity: .9; }
.send-btn:active:not(:disabled) { transform: scale(.97); }
.send-btn:disabled { background: #c7c9d1; cursor: not-allowed; }

/* 动画 */
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes blink { 50% { opacity: 0; } }

/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
</style>

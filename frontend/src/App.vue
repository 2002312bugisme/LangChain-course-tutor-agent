<script setup>
// 课栈 - 编程学习助手（阶段 4 + 6.5）
// 阶段 6.5：侧边栏会话列表——新建/切换会话、加载历史
import { ref, onMounted } from 'vue'

const messages = ref([])        // {role: 'user'|'assistant', content, thinking, tools: []}
const input = ref('')
const loading = ref(false)
const chatBox = ref(null)
const threads = ref([])         // 会话列表 [{thread_id, step, ts, title}]
const activeThread = ref('')
const editingId = ref(null)      // 正在重命名的会话 id
const editingTitle = ref('')

// thread_id 会话管理（localStorage 持久化，刷新页面续聊）
const THREAD_KEY = 'kezhan_thread_id'

function scrollBottom() {
  setTimeout(() => {
    if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
  }, 50)
}

// 加载会话列表（每次发送后刷新，保证最新会话在顶部）
async function loadThreads() {
  try {
    const resp = await fetch('/threads')
    if (resp.ok) threads.value = await resp.json()
  } catch (e) { /* 后端未就绪时忽略 */ }
}

// 切换会话：拉取该会话历史并渲染
async function switchThread(tid) {
  if (loading.value) return
  activeThread.value = tid
  localStorage.setItem(THREAD_KEY, tid)
  messages.value = []
  try {
    const resp = await fetch(`/threads/${encodeURIComponent(tid)}/messages`)
    if (resp.ok) {
      const data = await resp.json()
      messages.value = data.messages.map((m) => ({
        role: m.role === 'human' ? 'user' : 'assistant',
        content: m.content,
        thinking: '',
        tools: [],
      }))
      scrollBottom()
    }
  } catch (e) { /* ignore */ }
}

// 新建会话：新 thread_id + 清空消息 + 刷新列表
async function newSession() {
  if (loading.value) return
  activeThread.value = crypto.randomUUID()
  localStorage.setItem(THREAD_KEY, activeThread.value)
  messages.value = []
  await loadThreads()
}

// 重命名：进入编辑态
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

// 删除会话：确认后删除；若删的是当前会话则新建一个
async function removeThread(tid) {
  if (loading.value) return
  if (!confirm('确定删除该会话吗？删除后不可恢复。')) return
  try {
    await fetch(`/threads/${encodeURIComponent(tid)}`, { method: 'DELETE' })
    if (activeThread.value === tid) await newSession()
    else await loadThreads()
  } catch (e) { /* ignore */ }
}

async function send() {
  const text = input.value.trim()
  if (!text || loading.value) return
  input.value = ''
  loading.value = true

  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '', thinking: '', tools: [] })
  // ★ Vue 3 响应式陷阱修复：从代理数组取引用（阶段 4 踩坑）
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
        else if (evt.type === 'token') msg.content += evt.content
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
    await loadThreads()   // 发送完刷新会话列表（新会话/更新排序）
  }
}

onMounted(async () => {
  await loadThreads()
  const saved = localStorage.getItem(THREAD_KEY)
  if (saved && threads.value.some((t) => t.thread_id === saved)) {
    await switchThread(saved)   // 恢复上次会话及其历史
  } else {
    activeThread.value = crypto.randomUUID()
    localStorage.setItem(THREAD_KEY, activeThread.value)
  }
})
</script>

<template>
  <div class="layout">
    <!-- 侧边栏：新建会话 + 会话列表 -->
    <aside class="sidebar">
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
          <!-- 重命名编辑态 -->
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
            <div class="thread-title">{{ t.title }}</div>
            <div class="thread-actions" @click.stop>
              <button class="icon-btn" title="重命名" @click="startRename(t)">✏️</button>
              <button class="icon-btn" title="删除会话" @click="removeThread(t.thread_id)">🗑</button>
            </div>
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
      </header>

      <main ref="chatBox" class="chat">
        <div v-if="messages.length === 0" class="empty">
          问我课程相关的问题试试，比如：<br />
          "推荐一门 Python 入门课" · "有没有 Vue 进阶课程？"
        </div>

        <div v-for="(m, i) in messages" :key="i" class="row" :class="m.role">
          <div class="bubble">
            <template v-if="m.role === 'assistant'">
              <div v-if="m.thinking" class="thinking">
                <div class="label">🤔 思考过程</div>
                <div class="content">{{ m.thinking }}</div>
              </div>
              <div v-for="(t, j) in m.tools" :key="j" class="tool-call">
                🔧 调用工具 <b>{{ t.name }}</b> → {{ t.result }}
              </div>
              <div v-if="m.content" class="answer">{{ m.content }}</div>
              <div v-else-if="loading && i === messages.length - 1" class="typing">思考中…</div>
            </template>
            <template v-else>{{ m.content }}</template>
          </div>
        </div>
      </main>

      <footer>
        <input
          v-model="input"
          placeholder="输入你的问题…"
          :disabled="loading"
          @keyup.enter="send"
        />
        <button :disabled="loading || !input.trim()" @click="send">
          {{ loading ? '生成中…' : '发送' }}
        </button>
      </footer>
    </div>
  </div>
</template>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f0f2f5; }

/* 整体布局：侧边栏 + 聊天区 */
.layout { display: flex; height: 100vh; }
.sidebar { width: 240px; background: #1e293b; color: #e2e8f0; display: flex; flex-direction: column; flex-shrink: 0; }
.new-session { margin: 14px; padding: 10px; background: #2563eb; color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
.new-session:hover { background: #1d4ed8; }
.new-session:disabled { opacity: .5; cursor: not-allowed; }
.thread-list { flex: 1; overflow-y: auto; padding: 0 10px 14px; }
.thread-item { padding: 8px 10px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; gap: 6px; }
.thread-item:hover { background: #334155; }
.thread-item.active { background: #2563eb; }
.thread-title { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.thread-actions { display: none; gap: 2px; flex-shrink: 0; }
.thread-item:hover .thread-actions { display: flex; }
.icon-btn { background: none; border: none; cursor: pointer; font-size: 12px; padding: 2px 3px; border-radius: 4px; opacity: .8; }
.icon-btn:hover { background: rgba(255,255,255,.2); opacity: 1; }
.rename-input { width: 100%; padding: 4px 6px; border: 1px solid #60a5fa; border-radius: 4px; font-size: 13px; background: #fff; color: #1e293b; outline: none; }
.thread-step { font-size: 11px; color: #94a3b8; }
.thread-empty { color: #64748b; font-size: 13px; text-align: center; padding: 20px 0; }

/* 聊天区（沿用阶段 4 样式） */
.app { flex: 1; display: flex; flex-direction: column; min-width: 0; }
header { padding: 16px 20px; background: #fff; border-bottom: 1px solid #e5e7eb; display: flex; align-items: baseline; gap: 12px; }
header h1 { font-size: 20px; color: #1f2937; }
.hint { font-size: 12px; color: #9ca3af; }
.chat { flex: 1; overflow-y: auto; padding: 20px; }
.empty { text-align: center; color: #9ca3af; margin-top: 80px; line-height: 2; }
.row { display: flex; margin-bottom: 16px; }
.row.user { justify-content: flex-end; }
.row.user .bubble { background: #2563eb; color: #fff; }
.bubble { max-width: 78%; padding: 12px 16px; border-radius: 12px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
.thinking { background: #f8fafc; border-left: 3px solid #94a3b8; border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; }
.thinking .label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
.thinking .content { font-size: 13px; color: #64748b; }
.tool-call { font-size: 12px; color: #0369a1; background: #f0f9ff; border-radius: 6px; padding: 6px 10px; margin-bottom: 8px; }
.answer { font-size: 15px; color: #1f2937; }
.typing { color: #9ca3af; font-size: 14px; }
footer { display: flex; gap: 10px; padding: 14px 20px; background: #fff; border-top: 1px solid #e5e7eb; }
footer input { flex: 1; padding: 10px 14px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; outline: none; }
footer input:focus { border-color: #2563eb; }
footer button { padding: 10px 24px; background: #2563eb; color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }
footer button:disabled { background: #93c5fd; cursor: not-allowed; }
</style>

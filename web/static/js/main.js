// 全局变量
const API_BASE_URL = '';

// 工具函数
function showLoading(element) {
    element.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <p class="mt-2 text-muted">正在加载数据...</p>
        </div>
    `;
}

function clearElement(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

function appendIcon(parent, className) {
    const icon = document.createElement('i');
    icon.className = className;
    parent.appendChild(icon);
}

function appendTextAlert(element, type, iconClass, message) {
    clearElement(element);
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    appendIcon(alertDiv, iconClass);
    alertDiv.appendChild(document.createTextNode(' ' + String(message || '')));
    element.appendChild(alertDiv);
}

function showError(element, message) {
    appendTextAlert(element, 'danger', 'bi bi-exclamation-triangle', message);
}

function showSuccess(element, message) {
    appendTextAlert(element, 'success', 'bi bi-check-circle', message);
}

// 格式化数字
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    const parsed = parseFloat(num);
    return Number.isFinite(parsed) ? parsed.toFixed(decimals) : '-';
}

function formatPercent(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    const parsed = parseFloat(num);
    return Number.isFinite(parsed) ? (parsed * 100).toFixed(decimals) + '%' : '-';
}

// 显示通知
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.appendChild(document.createTextNode(String(message || '')));
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close';
    closeButton.setAttribute('data-bs-dismiss', 'alert');
    alertDiv.appendChild(closeButton);

    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => alertDiv.remove(), 5000);
    }
}

// API调用函数
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API调用失败:', error);
        throw error;
    }
}

// 任务状态轮询
async function pollTaskStatus(taskId, callback, interval = 1000) {
    const maxAttempts = 300; // 5分钟
    let attempts = 0;

    const checkStatus = async () => {
        try {
            const data = await apiCall(`/api/task/${taskId}`);

            if (data.state === 'SUCCESS') {
                callback(null, data.result);
            } else if (data.state === 'FAILURE') {
                callback(new Error(data.error || '任务执行失败'));
            } else if (attempts < maxAttempts) {
                attempts++;
                setTimeout(checkStatus, interval);
            } else {
                callback(new Error('任务超时'));
            }
        } catch (error) {
            callback(error);
        }
    };

    checkStatus();
}

// 股票分析
async function analyzeStock(stockCode, strategyType = 'technical') {
    try {
        showNotification('正在提交分析任务...', 'info');

        const data = await apiCall('/api/analyze', {
            method: 'POST',
            body: JSON.stringify({
                stock_code: stockCode,
                strategy_type: strategyType
            })
        });

        if (data.error) {
            throw new Error(data.error);
        }

        showNotification('分析任务已提交，正在处理...', 'success');

        // 开始轮询任务状态
        pollTaskStatus(data.task_id, (error, result) => {
            if (error) {
                showNotification(`分析失败: ${error.message}`, 'danger');
            } else {
                showNotification('分析完成！', 'success');
                displayAnalysisResult(result);
            }
        });

    } catch (error) {
        showNotification(`分析请求失败: ${error.message}`, 'danger');
    }
}

// 创建回测
async function createBacktest(stockCode, strategyId, startDate, endDate, initialCapital = 100000) {
    try {
        showNotification('正在创建回测任务...', 'info');

        const data = await apiCall('/api/backtest', {
            method: 'POST',
            body: JSON.stringify({
                stock_code: stockCode,
                strategy_id: strategyId,
                start_date: startDate,
                end_date: endDate,
                initial_capital: initialCapital
            })
        });

        if (data.error) {
            throw new Error(data.error);
        }

        showNotification('回测任务已创建，正在处理...', 'success');

        // 开始轮询任务状态
        pollTaskStatus(data.task_id, (error, result) => {
            if (error) {
                showNotification(`回测失败: ${error.message}`, 'danger');
            } else {
                showNotification('回测完成！', 'success');
                // 刷新页面或更新结果
                location.reload();
            }
        });

    } catch (error) {
        showNotification(`回测请求失败: ${error.message}`, 'danger');
    }
}

// 显示分析结果
function displayAnalysisResult(result) {
    const modal = document.getElementById('analysis-result-modal');
    if (!modal) return;

    const content = modal.querySelector('.modal-body');
    if (!content) return;

    let html = '<div class="row g-4">';

    // 基本信息
    html += '<div class="col-md-6">';
    html += '<h6 class="fw-bold mb-3 text-primary"><i class="bi bi-info-circle me-2"></i>基本信息</h6>';
    html += '<div class="card border-0 bg-light"><div class="card-body p-3">';
    html += '<table class="table table-sm table-borderless mb-0">';
    html += '<tr><td class="text-muted">综合评分</td><td class="text-end"><span class="fw-bold text-primary h5 mb-0">' + formatNumber(result.total_score) + '</span></td></tr>';

    let badgeClass = 'secondary';
    let badgeText = 'N/A';
    if (result.recommendation === 'buy') { badgeClass = 'success'; badgeText = '买入'; }
    else if (result.recommendation === 'sell') { badgeClass = 'danger'; badgeText = '卖出'; }
    else if (result.recommendation === 'hold') { badgeClass = 'warning'; badgeText = '持有'; }

    html += '<tr><td class="text-muted">推荐操作</td><td class="text-end"><span class="badge bg-' + badgeClass + ' bg-opacity-10 text-' + badgeClass + ' px-3 py-2 rounded-pill">' + badgeText + '</span></td></tr>';
    html += '</table></div></div>';
    html += '</div>';

    // 技术指标
    html += '<div class="col-md-6">';
    html += '<h6 class="fw-bold mb-3 text-success"><i class="bi bi-graph-up me-2"></i>技术指标</h6>';
    html += '<div class="card border-0 bg-light"><div class="card-body p-3">';
    html += '<table class="table table-sm table-borderless mb-0">';
    html += '<tr><td class="text-muted">RSI</td><td class="text-end fw-medium">' + formatNumber(result.rsi) + '</td></tr>';
    html += '<tr><td class="text-muted">MACD</td><td class="text-end fw-medium">' + formatNumber(result.macd) + '</td></tr>';
    html += '<tr><td class="text-muted">爆发潜力</td><td class="text-end fw-medium">' + formatNumber(result.explosion_potential) + '</td></tr>';
    html += '</table></div></div>';
    html += '</div>';

    html += '</div>';

    content.innerHTML = html;

    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// 创建图表
function createChart(containerId, option) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    const chart = echarts.init(container);
    chart.setOption(option);

    // 响应式调整
    window.addEventListener('resize', () => chart.resize());

    return chart;
}

// 创建K线图
function createKLineChart(containerId, data) {
    const option = {
        title: {
            text: 'K线图',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        legend: {
            data: ['K线', 'MA5', 'MA10', 'MA20'],
            top: 30
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: data.dates,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false },
            min: 'dataMin',
            max: 'dataMax'
        },
        yAxis: {
            scale: true,
            splitArea: {
                show: true
            }
        },
        dataZoom: [
            {
                type: 'inside',
                start: 50,
                end: 100
            },
            {
                show: true,
                type: 'slider',
                top: '90%',
                start: 50,
                end: 100
            }
        ],
        series: [
            {
                name: 'K线',
                type: 'candlestick',
                data: data.values,
                itemStyle: {
                    color: '#c23531',
                    color0: '#314656',
                    borderColor: '#c23531',
                    borderColor0: '#314656'
                }
            },
            {
                name: 'MA5',
                type: 'line',
                data: data.ma5,
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            },
            {
                name: 'MA10',
                type: 'line',
                data: data.ma10,
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            },
            {
                name: 'MA20',
                type: 'line',
                data: data.ma20,
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            }
        ]
    };

    return createChart(containerId, option);
}

// 创建回测结果图表
function createBacktestChart(containerId, data) {
    const option = {
        title: {
            text: '回测结果',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis'
        },
        legend: {
            data: ['资产净值', '基准'],
            top: 30
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: data.dates
        },
        yAxis: {
            type: 'value',
            axisLabel: {
                formatter: '{value}%'
            }
        },
        series: [
            {
                name: '资产净值',
                type: 'line',
                data: data.portfolio,
                smooth: true,
                lineStyle: {
                    width: 2
                }
            },
            {
                name: '基准',
                type: 'line',
                data: data.benchmark,
                smooth: true,
                lineStyle: {
                    width: 2,
                    type: 'dashed'
                }
            }
        ]
    };

    return createChart(containerId, option);
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 初始化弹出框
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // 添加加载动画到所有表单提交
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>处理中...';
            }
        });
    });
});

// 导出函数供全局使用
window.QSSS = {
    analyzeStock,
    createBacktest,
    createChart,
    createKLineChart,
    createBacktestChart,
    showNotification,
    formatNumber,
    formatPercent
};

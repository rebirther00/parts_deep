'use strict';
// This separate, small script also reports a failure to load/parse the main UI.
window.door8BootSettled = false;
window.door8StartupError = function(message) {
  const banner = document.getElementById('startup-error');
  banner.textContent = message;
  banner.hidden = false;
  document.getElementById('progress').textContent = '연결 상태 확인 필요';
};
window.addEventListener('error', function(event) {
  if (window.door8BootSettled) return;
  window.door8StartupError('화면을 시작하지 못했습니다. ' + (event.message || '스크립트를 불러오지 못했습니다.') + ' 새로고침해 주세요.');
}, true);
window.addEventListener('unhandledrejection', function(event) {
  if (window.door8BootSettled) return;
  const reason = event.reason && event.reason.message ? event.reason.message : '브라우저 초기화 오류';
  window.door8StartupError(reason + ' 서버를 재시작하고 새로고침해 주세요.');
});
setTimeout(function() {
  if (!window.door8BootSettled) {
    window.door8StartupError('화면 연결이 지연되고 있습니다. 서버 터미널이 실행 중인지와 8765 포트 전달을 확인한 뒤 새로고침해 주세요.');
  }
}, 10000);

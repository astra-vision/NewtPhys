window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function initMapOverlayPlayer() {
  var player = document.querySelector('[data-map-player]');
  if (!player) {
    return;
  }

  var playerVideo = document.getElementById('newtphys-map-video');
  var sceneSwitcher = player.querySelector('.map-scene-switcher');
  var buttons = Array.prototype.slice.call(player.querySelectorAll('.map-toggle-button'));
  var rgbButton = player.querySelector('.map-toggle-button[data-map-role="rgb"]') || buttons[0];
  var overlayButtons = buttons.filter(function(button) {
    return button !== rgbButton;
  });
  var manifestPath = player.dataset.manifestPath || '';
  var sceneButtons = [];
  var sceneRecordsByPath = {};

  if (!playerVideo || !buttons.length || !sceneSwitcher) {
    return;
  }

  var activeScenePath = playerVideo.dataset.scenePath || '';
  var activeOverlayButton = null;

  function buildSceneAssetPath(scenePath, fileName) {
    return scenePath + '/' + fileName;
  }

  function getSceneRecord(scenePath) {
    return sceneRecordsByPath[scenePath] || null;
  }

  function getButtonVideoFile(button) {
    return button ? (button.dataset.videoFile || '') : '';
  }

  function updateButtonState() {
    buttons.forEach(function(button) {
      var isActive = (button === rgbButton && !button.disabled) || button === activeOverlayButton;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  function updateSceneButtonState(activeButton) {
    sceneButtons.forEach(function(button) {
      var isActive = button === activeButton;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  function updateOverlayAvailability() {
    var sceneRecord = getSceneRecord(activeScenePath);

    buttons.forEach(function(button) {
      var fileName = getButtonVideoFile(button);
      var isEnabled = true;
      var disabledReason = '';
      var hasAvailabilityData = sceneRecord && sceneRecord.available_files && sceneRecord.available_files.length;

      if (hasAvailabilityData) {
        isEnabled = sceneRecord.available_files.indexOf(fileName) !== -1;
        disabledReason = (sceneRecord.disabled_files && sceneRecord.disabled_files[fileName]) || '';
      }

      button.classList.toggle('is-disabled', !isEnabled);
      button.disabled = !isEnabled;
      button.setAttribute('aria-disabled', isEnabled ? 'false' : 'true');

      if (disabledReason) {
        button.title = disabledReason;
      } else {
        button.removeAttribute('title');
      }
    });

    if (activeOverlayButton && activeOverlayButton.disabled) {
      activeOverlayButton = null;
    }

    if ((!rgbButton || rgbButton.disabled) && !activeOverlayButton) {
      activeOverlayButton = getFirstAvailableOverlayButton();
    }

    updateButtonState();
  }

  function getFirstAvailableButton(candidates) {
    for (var i = 0; i < candidates.length; i++) {
      if (!candidates[i].disabled) {
        return candidates[i];
      }
    }

    return null;
  }

  function getFirstAvailableOverlayButton() {
    return getFirstAvailableButton(overlayButtons);
  }

  function getFallbackButton() {
    if (rgbButton && !rgbButton.disabled) {
      return rgbButton;
    }

    return getFirstAvailableOverlayButton();
  }

  function getCurrentSourceButton() {
    if (activeOverlayButton && !activeOverlayButton.disabled) {
      return activeOverlayButton;
    }

    return getFallbackButton();
  }

  function getCurrentVideoFile() {
    return getButtonVideoFile(getCurrentSourceButton());
  }

  function getCurrentVideoSource() {
    var currentVideoFile = getCurrentVideoFile();
    if (!activeScenePath || !currentVideoFile) {
      return '';
    }

    return buildSceneAssetPath(activeScenePath, currentVideoFile);
  }

  function setVideoSource(nextSource, shouldAutoplay) {
    if (!nextSource || !activeScenePath) {
      return;
    }

    var normalizedNextSource = nextSource.replace(/^\.\//, '');
    var currentSource = playerVideo.currentSrc || '';
    playerVideo.dataset.shouldAutoplay = shouldAutoplay ? 'true' : 'false';

    if (currentSource.endsWith(normalizedNextSource)) {
      if (shouldAutoplay) {
        var resumedPlayPromise = playerVideo.play();
        if (resumedPlayPromise && typeof resumedPlayPromise.catch === 'function') {
          resumedPlayPromise.catch(function() {});
        }
      }
      return;
    }

    playerVideo.src = nextSource;
    playerVideo.load();

    if (shouldAutoplay) {
      var playPromise = playerVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(function() {});
      }
    }
  }

  playerVideo.addEventListener('error', function() {
    var shouldAutoplay = playerVideo.dataset.shouldAutoplay === 'true';
    var fallbackButton = getFallbackButton();
    var fallbackFile = getButtonVideoFile(fallbackButton);
    var fallbackSource = fallbackFile ? buildSceneAssetPath(activeScenePath, fallbackFile) : '';
    var normalizedFallbackSource = fallbackSource.replace(/^\.\//, '');
    var currentSource = playerVideo.currentSrc || playerVideo.src || '';

    if (!fallbackSource || currentSource.endsWith(normalizedFallbackSource)) {
      return;
    }

    activeOverlayButton = fallbackButton && fallbackButton !== rgbButton ? fallbackButton : null;
    playerVideo.dataset.videoFile = getCurrentVideoFile();
    updateButtonState();
    playerVideo.src = fallbackSource;
    playerVideo.load();

    if (shouldAutoplay) {
      var playPromise = playerVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(function() {});
      }
    }
  });

  function activateOverlay(button) {
    if (!button || button.disabled) {
      return;
    }

    activeOverlayButton = button;
    updateButtonState();
    playerVideo.dataset.videoFile = getCurrentVideoFile();
    setVideoSource(getCurrentVideoSource(), !playerVideo.paused);
  }

  function deactivateOverlay() {
    activeOverlayButton = null;
    updateButtonState();
    playerVideo.dataset.videoFile = getCurrentVideoFile();
    setVideoSource(getCurrentVideoSource(), !playerVideo.paused);
  }

  function activateScene(button) {
    updateSceneButtonState(button);
    activeScenePath = button.dataset.scenePath || activeScenePath;
    playerVideo.dataset.scenePath = activeScenePath;
    updateOverlayAvailability();
    playerVideo.dataset.videoFile = getCurrentVideoFile();
    setVideoSource(getCurrentVideoSource(), !playerVideo.paused);
  }

  function refreshSceneButtons() {
    sceneButtons = Array.prototype.slice.call(sceneSwitcher.querySelectorAll('.map-scene-button'));
  }

  function bindSceneButtons() {
    sceneButtons.forEach(function(button) {
      button.addEventListener('click', function() {
        activateScene(button);
      });
    });
  }

  function buildSceneButtons(sceneRecords, activePath) {
    Array.prototype.slice.call(sceneSwitcher.querySelectorAll('.map-scene-button')).forEach(function(button) {
      button.remove();
    });

    sceneRecords.forEach(function(record) {
      var button = document.createElement('button');
      var isActive = record.path === activePath;

      button.className = isActive ? 'map-scene-button is-active' : 'map-scene-button';
      button.type = 'button';
      button.dataset.scenePath = record.path;
      button.dataset.availableVideos = (record.available_files || []).join(',');
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      button.textContent = record.label;
      sceneSwitcher.appendChild(button);
    });

    refreshSceneButtons();
    bindSceneButtons();
  }

  function primeSceneRecordsFromButtons() {
    sceneRecordsByPath = {};
    refreshSceneButtons();
    sceneButtons.forEach(function(button) {
      var scenePath = button.dataset.scenePath || '';
      if (!scenePath) {
        return;
      }

      var availableFiles = (button.dataset.availableVideos || '')
        .split(',')
        .map(function(fileName) {
          return fileName.trim();
        })
        .filter(function(fileName) {
          return Boolean(fileName);
        });
      var disabledFiles = {};

      buttons.forEach(function(toggleButton) {
        var fileName = getButtonVideoFile(toggleButton);
        if (fileName && availableFiles.length && availableFiles.indexOf(fileName) === -1) {
          disabledFiles[fileName] = 'Not available for this scene.';
        }
      });

      sceneRecordsByPath[scenePath] = {
        label: button.textContent.trim(),
        path: scenePath,
        available_files: availableFiles.length ? availableFiles : null,
        disabled_files: disabledFiles
      };
    });
  }

  function applyManifest(manifest) {
    if (!manifest || !manifest.scenes || !manifest.scenes.length) {
      return;
    }

    var nextActivePath = activeScenePath;
    sceneRecordsByPath = {};

    manifest.scenes.forEach(function(record) {
      sceneRecordsByPath[record.path] = record;
    });

    if (!sceneRecordsByPath[nextActivePath]) {
      nextActivePath = manifest.scenes[0].path;
    }

    buildSceneButtons(manifest.scenes, nextActivePath);

    var initialSceneButton = sceneButtons[0];
    for (var i = 0; i < sceneButtons.length; i++) {
      if (sceneButtons[i].dataset.scenePath === nextActivePath) {
        initialSceneButton = sceneButtons[i];
        break;
      }
    }

    if (initialSceneButton) {
      activeScenePath = initialSceneButton.dataset.scenePath || nextActivePath;
      playerVideo.dataset.scenePath = activeScenePath;
      updateSceneButtonState(initialSceneButton);
      updateOverlayAvailability();
      playerVideo.dataset.videoFile = getCurrentVideoFile();
      setVideoSource(getCurrentVideoSource(), !playerVideo.paused);
    }
  }

  function getActiveOverlayButton() {
    for (var i = 0; i < overlayButtons.length; i++) {
      if (overlayButtons[i].classList.contains('is-active') && !overlayButtons[i].disabled) {
        return overlayButtons[i];
      }
    }

    return null;
  }

  buttons.forEach(function(button) {
    button.addEventListener('click', function() {
      if (button.disabled) {
        return;
      }

      if (button === rgbButton) {
        deactivateOverlay();
        return;
      }

      if (button === activeOverlayButton) {
        if (!rgbButton || rgbButton.disabled) {
          return;
        }

        deactivateOverlay();
        return;
      }

      activateOverlay(button);
    });
  });

  primeSceneRecordsFromButtons();
  bindSceneButtons();

  var initialSceneButton = player.querySelector('.map-scene-button.is-active') || sceneButtons[0];
  if (initialSceneButton) {
    updateSceneButtonState(initialSceneButton);
    activeScenePath = initialSceneButton.dataset.scenePath || '';
  }
  playerVideo.dataset.scenePath = activeScenePath;
  activeOverlayButton = getActiveOverlayButton();
  updateOverlayAvailability();
  playerVideo.dataset.videoFile = getCurrentVideoFile();
  setVideoSource(getCurrentVideoSource(), !playerVideo.paused);

  if (window.fetch && manifestPath) {
    fetch(manifestPath, {cache: 'no-store'})
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Failed to load map manifest.');
        }
        return response.json();
      })
      .then(function(manifest) {
        applyManifest(manifest);
      })
      .catch(function() {});
  }
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for (var i = 0; i < carousels.length; i++) {
      carousels[i].on('before:show', function() {});
    }

    var interpolationSlider = $('#interpolation-slider');
    var interpolationWrapper = $('#interpolation-image-wrapper');
    if (interpolationSlider.length && interpolationWrapper.length) {
      preloadInterpolationImages();

      interpolationSlider.on('input', function() {
        setInterpolationImage(this.value);
      });
      setInterpolationImage(0);
      interpolationSlider.prop('max', NUM_INTERP_FRAMES - 1);
    }

    bulmaSlider.attach();
    initMapOverlayPlayer();

})
